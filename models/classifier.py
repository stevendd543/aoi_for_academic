
import torch
import torch.nn as nn
import torchvision.models as models


def create_network(model_type='mobilenet_v2', pretrained=False):
    """
    model_type :
    - 'mobilenet_v2': MobileNetV2 (3.4M)
    - 'mobilenet_v3_small': MobileNetV3 Small (2.5M) 
    - 'mobilenet_v3_large': MobileNetV3 Large (5.4M)
    - 'efficientnet_b0': EfficientNet-B0 (5.3M)
    - 'shufflenet_v2_x0_5': ShuffleNetV2 0.5x (1.4M)
    - 'shufflenet_v2_x1_0': ShuffleNetV2 1.0x (2.3M)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(960, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    
    elif model_type == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        model.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = (
        f"Model: {model_type}\n"
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}\n"
        f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB\n"
    )
    
    return model, device, model_info