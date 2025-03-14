import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    VGG16_Weights,
    VGG19_Weights,
    ResNet50_Weights,
    DenseNet121_Weights,
    DenseNet201_Weights,
    MobileNet_V2_Weights,
    Inception_V3_Weights,
    EfficientNet_B0_Weights
)

class CNNPoseClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNPoseClassifier, self).__init__()
        self.model_name = model_name
        
        # Initialize the selected model
        if model_name == 'vgg16':
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vgg16(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        elif model_name == 'vgg19':
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vgg19(weights=weights)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.model.fc = nn.Linear(2048, 2)
        elif model_name == 'densenet121':
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.densenet121(weights=weights)
            self.model.classifier = nn.Linear(1024, 2)
        elif model_name == 'densenet201':
            weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.densenet201(weights=weights)
            self.model.classifier = nn.Linear(1920, 2)
        elif model_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            self.model.classifier[-1] = nn.Linear(1280, 2)
        elif model_name == 'inception_v3':
            weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.inception_v3(weights=weights)
            self.model.fc = nn.Linear(2048, 2)
            self.model.aux_logits = False
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            self.model.classifier[-1] = nn.Linear(1280, 2)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def forward(self, x):
        return self.model(x)
