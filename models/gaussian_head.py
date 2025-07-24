# models/gaussian_head.py

import torch
import torch.nn as nn

class GaussianParamHead(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
        )
        # Output heads
        self.scale_head = nn.Conv2d(128, 3, 1)   # (sx, sy, sz)
        self.color_head = nn.Conv2d(128, 3, 1)   # (r, g, b)
        self.opacity_head = nn.Conv2d(128, 1, 1) # (alpha)
        # Optional: rotation_head = nn.Conv2d(128, 3, 1) or 6D

    def forward(self, features):
        x = self.net(features)
        scale = torch.sigmoid(self.scale_head(x)) * 0.5  # scale range: (0, 0.5)
        color = torch.sigmoid(self.color_head(x))        # RGB range: (0, 1)
        alpha = torch.sigmoid(self.opacity_head(x))      # Opacity: (0, 1)
        return {
            'scale': scale,
            'color': color,
            'opacity': alpha
        }
