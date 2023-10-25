# Hybrid R2AU-Net
Hybrid Attention Recurrent Residual U-Net for the Segmentation of Ultrasound Spine Images.

## Paper


## Dataset
The Hybrid R2AU-Net was designed to perform the segmentation of ultrasound spine images. The ultrasound spine image dataset is not publicly available but the Hybrid R2AU-Net can also be used for other imaging datasets if the following recommendations are followed:
- The full dataset is allocated to the "data"  folder.
- The input size is changed according to the dataset dimensions.

## The Hybrid R2AU-Net architecture
The Hybrid R2AU-Net adopts the foundational U-Net framework, renowned for its effectiveness in segmentation tasks. However, several enhancements have been introduced to elevate its performance. These modifications include the integration of attention mechanisms and RecRes Blocks, strategically positioned to capture and leverage spatial dependencies within the image data. Furthermore, dense connections were introduced to bridge the semantic gap between the encoder and decoder. The architecture of the proposed model is shown below.

![alt text](/images/Hybrid.png)

The Hybrid R2AU-Net adopts Recurrent Residual Blocks (RecRes Blocks), replacing the convolution blocks found in the original U-Net architecture. These blocks, illustrated below, incorporate both recurrent layers and residual connections to enhance segmentation quality.

![alt text](/images/RecResBlock.png)


## Additional Information
The Hybrid R2AU-Net was built with Python programming using Tensorflow (version 2.4.0) and Keras (version 2.10.0) frameworks.
Some functions used for the evaluation of the Hybrid R2AU-Net were inspired by the MultiResUNet repository - [See this GitHub Page](https://github.com/nibtehaz/MultiResUNet/tree/master).
More information about the model's architecture can be found in the paper.
