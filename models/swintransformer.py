""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Tuple, Type
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# def drop_path_f(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.

#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size, W // window_size, window_size, C
    )
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_partition_cca(x: torch.Tensor, window_size: int, top_padding: bool = False) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        if top_padding:
            x = F.pad(x, (0, 0, pad_w, 0, pad_h, 0))
        else:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition_cca(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer(
            "relative_position_index", relative_position_index
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(
            0
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        except Exception as e:
            attn = attn.to(dtype=x.dtype)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        c2,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        assert dim == c2, r"no. in/out channel should be same"
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i]
                        if isinstance(drop_path, list)
                        else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(c2, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros(
            (1, Hp, Wp, 1), device=x.device
        )  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, in_c=3, embed_dim=96, patch_size=4, stride=4, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride=(stride, stride)
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (
            W % self.patch_size[1] != 0
        )
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(
                x,
                (
                    0,
                    self.patch_size[1] - W % self.patch_size[1],
                    0,
                    self.patch_size[0] - H % self.patch_size[0],
                    0,
                    0,
                ),
            )

        x = self.proj(x)
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # view: [B, HW, C] -> [B, H, W, C]
        # permute: [B, H, W, C] -> [B, C, H, W]
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class PatchEmbed_CCA(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),                         #* changed stride from 16, 16  to 8 x 8
        padding: Tuple[int, int] = (1, 1),                          #* originally 0, 0padding should be 4, 4 in case stride is 8 , 8
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x

class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, c2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert c2 == (
            2 * dim
        ), r"no. out channel should be 2 * no. in channel "
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(B, H*W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        x = x.view(B, int(H / 2), int(W / 2), C * 2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ---------------------CCA----------------------------
# ---------------------- Cross Channel Attention ----------------------
class CCAttentionBlock(nn.Module):
    """Multi-modal transformer block for RGB+ channel attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        out_dim: int = 192,
        activation: Type[nn.Module] = nn.ReLU,
        skip_pe: bool = True,
        shift_size=0,
    ) -> None:
        super().__init__()

        self.r2g_attn = CCAttention(embedding_dim, num_heads, shift_size)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.rg2b_attn = CCAttention(embedding_dim, num_heads, shift_size)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.rgb2ir_attn = CCAttention(embedding_dim, num_heads, shift_size)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.ir2rgb_attn = CCAttention(embedding_dim, num_heads, shift_size)
        self.norm4 = nn.LayerNorm(embedding_dim)

        # self.mlp = Mlp(out_dim, out_dim * 4, out_dim, linear_mlp=True)

        self.window_size = 1
        self.input_resolution = (128, 128)
        self.shift_size = shift_size
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows, _ = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        # self.fc_layer = nn.Linear (192, out_dim)
        # #self.dropout = nn.Dropout(0.4)
        # self.mlp = MLPBlock(out_dim, 256, activation)
        # self.norm5 = nn.LayerNorm(out_dim)

    def forward(
        self,
        r: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        ir: torch.Tensor = None,
        window_size: int = 1,
        mode: str = "RGB+IR",
    ):
        b1, h, w, c = r.shape
        if mode == "RGB":
            ir = torch.randn(b1, h, w, c)
            ir = ir.cuda()
        # r, r_hw =window_partition(r, window_size)
        # g, g_hw = window_partition(g, window_size)
        # b, b_hw = window_partition(b, window_size)
        # ir, ir_hw = window_partition(ir, window_size)
        # b2, h2, w2, c2 = r.shape

        if self.shift_size > 0:
            # shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_r = torch.roll(
                r, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_g = torch.roll(
                g, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_b = torch.roll(
                b, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_ir = torch.roll(
                ir, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

            # partition of windows
            r_w, r_hw = window_partition_cca(shifted_r, self.window_size)
            g_w, g_hw = window_partition_cca(shifted_g, self.window_size)
            b_w, b_hw = window_partition_cca(shifted_b, self.window_size)
            ir_w, ir_hw = window_partition_cca(shifted_ir, self.window_size)
            b2, h2, w2, c2 = r_w.shape
            # partition windows
            # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        else:
            # shifted_x = x
            # partition windows
            # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            r_w, r_hw = window_partition_cca(r, self.window_size)
            g_w, g_hw = window_partition_cca(g, self.window_size)
            b_w, b_hw = window_partition_cca(b, self.window_size)
            ir_w, ir_hw = window_partition_cca(ir, self.window_size)
            b2, h2, w2, c2 = r_w.shape
        r_w = r_w.reshape(b2, h2 * w2, c2)
        g_w = g_w.reshape(b2, h2 * w2, c2)
        b_w = b_w.reshape(b2, h2 * w2, c2)
        ir_w = ir_w.reshape(b2, h2 * w2, c2)

        r_out = self.r2g_attn(
            q=r_w, k=g_w, v=g_w, dimensions=(h, w), mask=self.attn_mask
        )
        # x1 = r + r_out
        # x1 = self.norm1(x1)

        g_out = self.rg2b_attn(
            q=g_w, k=b_w, v=b_w, dimensions=(h, w), mask=self.attn_mask
        )
        # x2 = g + g_out
        # x2 = self.norm2(x2)

        b_out = self.rgb2ir_attn(
            q=b_w, k=ir_w, v=ir_w, dimensions=(h, w), mask=self.attn_mask
        )
        # x3 = b + b_out
        # x3 = self.norm3(x3)

        ir_out = self.ir2rgb_attn(
            q=ir_w, k=g_w, v=g_w, dimensions=(h, w), mask=self.attn_mask
        )
        # x4 = ir + ir_out
        # x4 = self.norm4(x4)

        r_out = r_out.view(b2, h2, w2, c2)
        g_out = g_out.view(b2, h2, w2, c2)
        b_out = b_out.view(b2, h2, w2, c2)
        ir_out = ir_out.view(b2, h2, w2, c2)
        if self.shift_size > 0:
            shifted_r = window_unpartition_cca(
                r_out, self.window_size, r_hw, (h, w)
            )  # B H' W' C
            shifted_g = window_unpartition_cca(
                g_out, self.window_size, g_hw, (h, w)
            )
            shifted_b = window_unpartition_cca(
                b_out, self.window_size, b_hw, (h, w)
            )
            shifted_ir = window_unpartition_cca(
                ir_out, self.window_size, ir_hw, (h, w)
            )
            r_out = torch.roll(
                shifted_r,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
            g_out = torch.roll(
                shifted_g,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
            b_out = torch.roll(
                shifted_b,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
            ir_out = torch.roll(
                shifted_ir,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            r_out = window_unpartition_cca(
                r_out, self.window_size, r_hw, (h, w)
            )  # B H' W' C

            g_out = window_unpartition_cca(g_out, self.window_size, g_hw, (h, w))
            b_out = window_unpartition_cca(b_out, self.window_size, b_hw, (h, w))
            ir_out = window_unpartition_cca(
                ir_out, self.window_size, ir_hw, (h, w)
            )
            # x = shifted_x

        x1 = self.norm1(r + r_out)
        x2 = self.norm2(g + g_out)
        x3 = self.norm3(b + b_out)
        x4 = self.norm4(ir + ir_out)

        # x = torch.cat((x1, x2, x3, x4), dim=-1)
        # x = self.mlp(x, h, w)
        # x = self.fc_layer(x)
        # #x = self.dropout(x)
        # x = self.mlp(x)
        # x = self.norm5(x)
        return x1, x2, x3, x4


class CCAttention(nn.Module):
    """attention layer without FFN allowing cross attention used for channels"""

    def __init__(
        self, embedding_dim: int, num_heads: int = 8, shift_size=0
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        # self.mlp = Mlp(embedding_dim, embedding_dim * 4, embedding_dim, linear_mlp=True)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dimensions,
        mask=None,
    ) -> torch.Tensor:
        """
        -dimensions: tuple(int, int) the original dimension of feature map before window partitioning in height and width
        """
        B_, N, C = q.shape
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # attn = self.softmax(attn)

            # attn = self.softmax(attn)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # output
        out = attn @ v
        out = self._recombine_heads(out)

        return out


class CrossChannelAttention(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        num_heads: int = 4,
        embed_dim: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.channel_embed_r = PatchEmbed_CCA(
            kernel_size=(1, 1),  #patch_size
            stride=(1, 1),         #* previoulsy 4  changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            padding = (0, 0),
            in_chans=1,
            embed_dim=embed_dim,
        )

        self.channel_embed_g = PatchEmbed_CCA(
            kernel_size=(1, 1),  #patch_size
            stride=(1, 1),         #* previoulsy 4  changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            padding = (0, 0),
            in_chans=1,
            embed_dim=embed_dim,
        )

        self.channel_embed_b = PatchEmbed_CCA(
            kernel_size=(1, 1),  #patch_size
            stride=(1, 1),         #* previoulsy 4  changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            padding = (0, 0),
            in_chans=1,
            embed_dim=embed_dim,
        )

        self.channel_embed_ir = PatchEmbed_CCA(
            kernel_size=(1, 1),  #patch_size
            stride=(1, 1),         #* previoulsy 4  changed from patch_size, patch_size to 8, 8 to get half-overlapping 64 x 64 dimension patches
            padding = (0, 0),
            in_chans=1,
            embed_dim=embed_dim,
        )

        self.chan_block = CCAttentionBlock(
            embedding_dim=1,
            num_heads=1,
        )
    
    def forward(self, img):
        r, g, b, ir = get_channels(img)
        r = self.channel_embed_r(r)
        g = self.channel_embed_g(g)
        b = self.channel_embed_b(b)
        ir = self.channel_embed_ir(ir)
        r, g, b, ir = self.chan_block(r, g, b, ir)
        x = torch.cat((r, g, b, ir), dim=-1)
        x = x.permute(0, 3, 2, 1).contiguous()

        return x


def get_channels(x):
    #idx = torch.tensor([0]).to(x.device)

    r = x[:,0,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,0,:,:]
    #idx = torch.tensor([1]).to(x.device)
    g = x[:,1,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,1,:,:].unsqueeze(1)
    #idx = torch.tensor([2]).to(x.device)
    b = x[:,2,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,2,:,:].unsqueeze(1)
    #idx = torch.tensor([3]).to(x.device)
    i = x[:,3,:,:].unsqueeze(1)#.index_select(1, idx)#.detach()#[:,3,:,:].unsqueeze(1)
    return r, g, b, i
    # return r, g, b