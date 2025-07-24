# utils/projection.py

import torch

def depth_to_3d(depth, K, R, T):
    """
    depth: [B, 1, H, W]
    K: [3, 3] intrinsic matrix
    R: [3, 3] rotation matrix
    T: [3, 1] translation vector
    returns: [B, 3, H, W] world coordinates
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Create meshgrid of pixel coordinates
    u = torch.arange(0, W, device=device)
    v = torch.arange(0, H, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # [H, W]

    # Convert to float
    grid_u = grid_u.float()
    grid_v = grid_v.float()

    # Stack into homogeneous pixel coordinates: [3, H*W]
    ones = torch.ones_like(grid_u)
    pixel_coords = torch.stack([grid_u, grid_v, ones], dim=0).reshape(3, -1)

    # Invert intrinsics
    K_inv = torch.inverse(K)

    # Backproject to camera coordinates: X_cam = K^-1 * [u,v,1] * depth
    cam_coords = K_inv @ pixel_coords  # [3, H*W]
    cam_coords = cam_coords.unsqueeze(0).repeat(B, 1, 1) * depth.view(B, 1, -1)  # [B, 3, H*W]

    # Transform to world coordinates: X_world = R * X_cam + T
    world_coords = R @ cam_coords + T

    return world_coords.view(B, 3, H, W)
