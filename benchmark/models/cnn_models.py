import torch
import torch.nn as nn
import torchvision.models as models

class CNNPoseClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNPoseClassifier, self).__init__()
        self.model_name = model_name
        
        # Initialize the selected model
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        elif model_name == 'vgg19':
            self.model = models.vgg19(pretrained=pretrained)
            self.model.classifier[-1] = nn.Linear(4096, 2)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(2048, 2)
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            self.model.classifier = nn.Linear(1024, 2)
        elif model_name == 'densenet201':
            self.model = models.densenet201(pretrained=pretrained)
            self.model.classifier = nn.Linear(1920, 2)
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.model.classifier[-1] = nn.Linear(1280, 2)
        elif model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=pretrained)
            self.model.fc = nn.Linear(2048, 2)
            self.model.aux_logits = False
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[-1] = nn.Linear(1280, 2)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def forward(self, x):
        return self.model(x)
