# models/depth_head.py
import torch.nn as nn
import torch

class DepthHead(nn.Module):
    def __init__(self, in_channels=512):
        super(DepthHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)  # Output depth
        )

    def forward(self, x):
        return torch.sigmoid(self.head(x))  # Normalize to (0, 1)
