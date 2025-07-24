# models/depth_net.py

import torch
import torch.nn as nn

class DepthNet(nn.Module):
    def __init__(self, in_channels=512):
        super(DepthNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),  # output depth: (B, 1, H, W)
            nn.Tanh()  # keep values between -1 and 1, later scaled to 0–1
        )

    def forward(self, x):
        return self.conv(x)

