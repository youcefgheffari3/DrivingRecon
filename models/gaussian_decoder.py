# models/gaussian_decoder.py
import torch.nn as nn
import torch

class GaussianDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super(GaussianDecoder, self).__init__()
        self.scale_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        scale = torch.sigmoid(self.scale_head(x))
        color = torch.sigmoid(self.color_head(x))
        opacity = torch.sigmoid(self.opacity_head(x))
        return scale, color, opacity
