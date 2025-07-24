# models/image_encoder.py

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(ImageEncoder, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # remove last FC layers

    def forward(self, x):
        return self.encoder(x)  # Output shape: (B, 512, H/32, W/32)
