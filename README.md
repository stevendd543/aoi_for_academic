# AOI detection system
This project provides tools for automated optical inspection (AOI) defect detection using deep learning models such as MobileNetV2, ResNet34, EfficientNet, and others.

## Features
 - Image segmentation and defect detection
 - Supports multiple model architectures
 - Easy inference and result saving

## Customization
To use other models, change model_type and weight_path in detect_test_image.py.

## Usage
- Prepare Model Weights
Place your trained model weights in the weights/ folder.
Example: weights/mobilenet_v2.pth

- Run Detection
Edit detect_test_image.py to set your model type, weight path, and image path.
