import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import torch.nn.functional as F

class SCFM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SCFM, self).__init__()
        mid_channels = max(1, channels // reduction_ratio)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel = self.channel_attention(x)
        x = x * channel
        spatial = torch.cat([torch.max(x, dim=1, keepdim=True)[0], 
                           torch.mean(x, dim=1, keepdim=True)], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial
        
        return x

class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)  
        return x

class Net(nn.Module):
    def __init__(self, encoder_name='timm-resnest200e', encoder_weights='imagenet', 
                 in_channels=3, classes=2):
        super(Net, self).__init__()
        self.PSPNet = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
        self.scfm = SCFM(channels=classes)
        self.aux_head = AuxiliaryHead(in_channels=512, num_classes=classes)
        
    def forward(self, x, return_features=False):
        features = self.PSPNet.encoder(x)
        aux_out = self.aux_head(features[-1])
        main_out = self.PSPNet(x)
        main_out = self.scfm(main_out)
        if return_features:
            return main_out, aux_out, features[-1]
        return main_out, aux_out