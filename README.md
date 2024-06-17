
The Swin-Transformer YOLOv5 model has been implemented and is available in [this GitHub repository](https://github.com/YJHCUI/SwinTransformer-YOLOv5)

# Swin-Transformer YOLOv5

This is a YOLOv5 code that incorporates SwinTransformer Blocks. It can be run in any environment capable of running YOLOv5 without requiring any additional libraries.

The code includes stages such as SwinTransformer Block, Patch Merging, and Patch Embed, making these modules adaptable to the model construction code of [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

Similar to YOLOv5, you can customize the model structure by modifying the model yaml file. For detailed instructions, refer to the [Usage Instructions](#usage-instructions).

## YOLOv5

The code is based on the [Ultralytics version of YOLOv5](https://github.com/ultralytics/yolov5). For more information about this YOLO repository, you can check the source code repository or its README file ([English](./.github/README.English.md) | [Simplified Chinese](./.github/README.zh-CN.md)).

## Swin Transformer

[Download Swin-Transformer paper](https://arxiv.org/abs/2103.14030)

<img src="./.github/pic/SwinTransformer.jpg" />

Swin-Transformer uses a hierarchical backbone network design. It starts with smaller patch sizes and merges patches at each stage, producing feature maps with different downsampling rates at different stages, thus outputting multi-scale feature information.

This hierarchical network structure that acquires multi-scale features makes SwinT more suitable for various CV tasks compared to ViT. Thanks to multi-scale features, the network can better perform detection, segmentation, and other dense detection tasks.

The SwinTransformer code used in this repository is from [this repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer). This code has detailed comments and includes padding for image and window size mismatches, making network design less restrictive, compared to the original [official code](https://github.com/microsoft/Swin-Transformer).

For Swin-Transformer applications in the R-CNN series, please refer to [this official repository](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

# Usage Instructions

## Overview

Data preparation and training methods are identical to YOLOv5's. For details, refer to [YOLOv5_Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

**The following mainly explains the model construction**

## Module Descriptions

The implementation code for Swin-Transformer-related modules is located in [swintransformer.py](./models/swintransformer.py) under the models folder. It mainly includes the SwinStage, Patch Merging, and PatchEmbed modules.

To ensure each module can be inserted into the original CNN structure to form a hybrid model, data shape transformations are added before and after each module, converting the [B, L, C] format used in Transformer Blocks to the [B, C, H, W] format suitable for CNNs.

### PatchEmbed

<img src="./.github/pic/PatchEmbed.jpg" />

This module is used at the beginning of the Backbone. In the PatchEmbed part, similar to ViT, it divides the image into fixed-size patches, flattens and linearly transforms each patch, and obtains the subsequent input vectors.

In this module, the following parameters need to be considered:

* `in_c=3`
  
  The number of channels in the input image. Typically, an RGB color image has 3 channels. **Each module**'s parameter in the network construction process will default to automatically select based on the input image size.

* `embed_dim=96`
  
  The length of the embedding vector, corresponding to the $C$ parameter in the paper. In a hybrid model, this step's final feature map channel count.
  
  SwinTransformer official model parameters are as follows:

  |Model|Size|C|
  |-----|----|--|
  |<table>...</table>|...

## Training Your Own Model

This step is consistent with the original YOLOv5 training method. When using `train.py`, set `--cfg` to point to your designed model's yaml file.
