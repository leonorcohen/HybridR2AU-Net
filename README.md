# Hybrid R2AU-Net
Hybrid Attention Recurrent Residual U-Net for the Segmentation of Ultrasound Spine Images.

## Paper


## Dataset
The Hybrid R2AU-Net was designed to perform the segmentation of ultrasound spine images. The ultrasound spine image dataset is not publicly available but the Hybrid can also be used for other imaging datasets if the following recommendations are followed:
- The full dataset is allocated to the "data"  folder.
- The input size is changed according to the dataset dimensions.

## The Hybrid R2AU-Net architecture
The Hybrid R2AU-Net adopts the foundational U-Net framework, renowned for its effectiveness in segmentation tasks. However, several enhancements have been introduced to elevate its performance. These modifications include the integration of attention mechanisms and RecRes Blocks, strategically positioned to capture and leverage spatial dependencies within the image data. Furthermore, dense connections were introduced to bridge the semantic gap between the encoder and decoder. The architecture of the proposed model is shown below.

![alt](/images/Hybrid.pdf)
