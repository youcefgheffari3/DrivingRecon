# run_temporal_drivingrecon.py

import torch
from loader.tum_loader import load_tum_sequence
from models.encoder import FeatureEncoder
from models.depth_head import DepthHead
from models.gaussian_decoder import GaussianDecoder
from models.pd_block import PD_Block
from utils.projection import depth_to_3d
from visualize_gaussians_panda import visualize_gaussians_panda
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
encoder = FeatureEncoder().to(device).eval()
depth_head = DepthHead().to(device).eval()
decoder = GaussianDecoder().to(device).eval()
pdblock = PD_Block().to(device).eval()

# === Load frames ===
frames = load_tum_sequence("data/fr1_desk", max_frames=10)
print(f"Loaded {len(frames)} frames.")

# === Camera intrinsics (use TUM fr1) ===
K = torch.tensor([
    [517.3, 0.0, 318.6],
    [0.0, 516.5, 255.3],
    [0.0, 0.0, 1.0]
], dtype=torch.float32).to(device)

K = K.unsqueeze(0)  # [1,3,3]

# === Accumulate all Gaussians here ===
all_gaussians = []

for idx, frame in enumerate(frames):
    print(f"\n▶️ Frame {idx + 1}")

    rgb = frame['rgb'].unsqueeze(0).to(device)     # [1,3,H,W]
    depth = frame['depth'].unsqueeze(0).to(device) # [1,1,H,W]
    R = frame['R'].unsqueeze(0).to(device)         # [1,3,3]
    T = frame['T'].unsqueeze(0).to(device)         # [1,3]

    features = encoder(rgb)
    depth_map = depth_head(features)

    world_coords = depth_to_3d(depth_map, K, R, T)  # [1,3,H,W]

    # Gaussian attributes
    scale = torch.sigmoid(torch.randn_like(world_coords))
    color = torch.sigmoid(rgb)
    opacity = torch.sigmoid(torch.randn(1, 1, *depth_map.shape[2:]).to(device))

    # PD-Block (simplified fusion)
    fused_feat = pdblock(features, features)

    # Predict Gaussian attributes
    scale_fused, color_fused, opacity_fused = decoder(fused_feat)

    # === Flatten Gaussians ===
    B, _, H, W = scale_fused.shape
    pos = world_coords.view(B, 3, -1).permute(0, 2, 1)     # [1, HW, 3]
    sca = scale_fused.view(B, 3, -1).permute(0, 2, 1)      # [1, HW, 3]
    col = color_fused.view(B, 3, -1).permute(0, 2, 1)      # [1, HW, 3]
    opa = opacity_fused.view(B, 1, -1).permute(0, 2, 1)    # [1, HW, 1]
    gaussians = torch.cat([pos, sca, col, opa], dim=-1)    # [1, HW, 10]

    all_gaussians.append(gaussians.detach().cpu())  # detach for safety

# === Combine all frames ===
final_gaussians = torch.cat(all_gaussians, dim=1)  # [1, N_total, 10]
print("✅ Final accumulated Gaussians:", final_gaussians.shape)

# === Save and visualize ===
np.savez("drivingrecon_final_gaussians.npz", gaussians=final_gaussians.numpy())
visualize_gaussians_panda(final_gaussians)
