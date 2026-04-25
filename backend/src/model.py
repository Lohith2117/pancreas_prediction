import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PancreaticDetectionNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Use the same weights we used in training
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        
        # Encoder Layers
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        
        # Classification Head (Trained for 1 output)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Linear(2048, 1)
        )
        
        # Segmentation Head (Matched to 256x256 resolution)
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        return self.cls_head(e4), self.seg_head(e4)