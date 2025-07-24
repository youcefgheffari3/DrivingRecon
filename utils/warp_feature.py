# utils/warp_feature.py

import torch
import torch.nn.functional as F
from utils.projection import depth_to_3d

def warp_feature(feat_i, depth_t, K, T_i_to_t):
    """
    Warps feat_i to align with current frame using depth and pose.

    feat_i: [B, C, H, W] — past frame feature
    depth_t: [B, 1, H, W] — depth of current frame
    K: [3, 3] intrinsics
    T_i_to_t: [B, 4, 4] relative transform from i → t
    """

    B, C, H, W = feat_i.shape
    device = feat_i.device

    # World coordinates of current frame
    R = torch.eye(3, device=device)
    T = torch.zeros(3, 1, device=device)
    world_coords = depth_to_3d(depth_t, K, R, T)  # [B, 3, H, W]

    # Transform world coords from t → i
    T_t_to_i = torch.inverse(T_i_to_t)  # [B, 4, 4]
    coords_flat = world_coords.view(B, 3, -1)  # [B, 3, HW]
    coords_flat = torch.cat([coords_flat, torch.ones(B, 1, H * W, device=device)], dim=1)  # [B, 4, HW]
    warped_coords = T_t_to_i @ coords_flat  # [B, 4, HW]
    warped_coords = warped_coords[:, :3, :]  # [B, 3, HW]

    # Project into image space
    pix_coords = K @ warped_coords  # [B, 3, HW]
    x = pix_coords[:, 0, :] / pix_coords[:, 2, :]
    y = pix_coords[:, 1, :] / pix_coords[:, 2, :]

    # Normalize to [-1, 1]
    x_norm = 2 * (x / (W - 1)) - 1
    y_norm = 2 * (y / (H - 1)) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).view(B, H, W, 2)

    # Sample from feature_i using the grid
    warped_feat = F.grid_sample(feat_i, grid, align_corners=True)
    return warped_feat
