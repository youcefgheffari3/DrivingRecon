import os
import torch
import torchvision.transforms as T
from PIL import Image
from models.image_encoder import ImageEncoder
from models.depth_net import DepthNet
from utils.projection import depth_to_3d
from models.gaussian_head import GaussianParamHead
from models.pd_block import PD_Block
from utils.warp_feature import warp_feature

# === Set your image path ===
image_path = r"C:\Users\Gheffari Youcef\Videos\DrivingRecon\data\Driving_in_Finland.jpg"

# === Load and preprocess image ===
if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
    exit(1)

image = Image.open(image_path).convert('RGB')

transform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(image).unsqueeze(0)

# === Feature Extraction ===
encoder = ImageEncoder()
encoder.eval()
features = encoder(img_tensor)

# === Predict Depth ===
depth_net = DepthNet()
depth_net.eval()
depth = depth_net(features)
depth = (depth + 1) / 2.0  # normalize to 0..1

print("✅ Feature map shape:", features.shape)
print("✅ Depth map shape:", depth.shape)
print("✅ Depth values (min/max):", depth.min().item(), "/", depth.max().item())

# === Convert depth to 3D coordinates ===
K = torch.tensor([[300.0, 0.0, 8.0],
                  [0.0, 300.0, 4.0],
                  [0.0, 0.0, 1.0]], dtype=torch.float32)
R = torch.eye(3, dtype=torch.float32)
T = torch.zeros((3, 1), dtype=torch.float32)
K, R, T = K.to(depth.device), R.to(depth.device), T.to(depth.device)

world_coords = depth_to_3d(depth, K, R, T)
print("✅ World coordinate shape:", world_coords.shape)
print("Sample (x, y, z) at center pixel:", world_coords[0, :, 4, 8])

# === Gaussian Parameters from Initial Features ===
gaussian_head = GaussianParamHead()
gaussian_head.eval()
params = gaussian_head(features)

print("✅ Scale shape:", params['scale'].shape)
print("✅ Color shape:", params['color'].shape)
print("✅ Opacity shape:", params['opacity'].shape)

# === Warp features and Fuse using PD-Block ===
T_i_to_t = torch.eye(4).unsqueeze(0).to(depth.device)  # identity
feat_i_warped = warp_feature(features, depth, K, T_i_to_t)

pdblock = PD_Block(in_channels=features.shape[1])
pdblock.eval()
fused_feat = pdblock(features, feat_i_warped)
print("✅ Fused feature shape:", fused_feat.shape)

# === Final Gaussian parameters ===
final_head = GaussianParamHead()
final_head.eval()
final_params = final_head(fused_feat)

scale_fused = final_params['scale']
color_fused = final_params['color']
opacity_fused = final_params['opacity']

print("✅ Final scale shape:", scale_fused.shape)
print("✅ Final color shape:", color_fused.shape)
print("✅ Final opacity shape:", opacity_fused.shape)

# === Flatten into final Gaussian representation ===
B, _, H, W = scale_fused.shape

positions = world_coords.view(B, 3, -1).permute(0, 2, 1)
scales    = scale_fused.view(B, 3, -1).permute(0, 2, 1)
colors    = color_fused.view(B, 3, -1).permute(0, 2, 1)
opacity   = opacity_fused.view(B, 1, -1).permute(0, 2, 1)

gaussians = torch.cat([positions, scales, colors, opacity], dim=-1)

print("✅ Final 3D Gaussians:", gaussians.shape)
print("Sample Gaussian (center):", gaussians[0, gaussians.shape[1] // 2])

# === Visualize with Panda3D ===
from visualize_gaussians_panda import visualize_gaussians_panda
torch.save(gaussians.detach(), "driving_scene.pt")
print("✅ Saved Gaussians to driving_scene.pt")
visualize_gaussians_panda(gaussians.detach())
