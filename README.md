# Dog vs Cat Classifier

This is a deep learning model that classifies images as either dogs or cats. The model is based on VGG16 architecture with transfer learning.

## Model Details
- Base model: VGG16 (pretrained on ImageNet)
- Input size: 150x150 RGB images
- Output: Binary classification (Dog vs Cat)

## How to Use
1. Upload an image of a dog or cat
2. The model will process the image and provide confidence scores for both classes
3. The prediction with the highest confidence will be shown first

## Technical Details
- The model uses transfer learning with VGG16 as the base model
- Additional dense layers were added for binary classification
- Images are preprocessed to 150x150 pixels and normalized before prediction

## Example Usage
You can use the example images provided or upload your own images of dogs or cats.