# models/pd_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PD_Block(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, 1)

        # Project feat_t to match hidden_dim before adding
        self.feat_t_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        # Final output projection back to in_channels
        self.fusion_conv = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, feat_t, feat_i):
        """
        feat_t: [B, C, H, W] — current frame features
        feat_i: [B, C, H, W] — aligned past frame features (warped to t)
        """

        # Create Q, K, V
        Q = self.query_conv(feat_t)         # [B, hidden, H, W]
        K = self.key_conv(feat_i)
        V = self.value_conv(feat_i)

        # Flatten
        B, C, H, W = Q.shape
        Q = Q.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        K = K.view(B, C, -1)                  # [B, C, HW]
        V = V.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # Attention
        attn = torch.bmm(Q, K) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        fused = torch.bmm(attn, V)  # [B, HW, C]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)

        # Project feat_t to same dim as fused
        feat_t_proj = self.feat_t_proj(feat_t)  # [B, hidden_dim, H, W]

        # Fuse
        out = self.fusion_conv(fused + feat_t_proj)
        return out
