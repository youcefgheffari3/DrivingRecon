# models/encoder.py
import torch.nn as nn
import torchvision.models as models

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool + fc

    def forward(self, x):
        return self.encoder(x)
