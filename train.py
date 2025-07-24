import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader.tum_loader import TUMSequenceDataset
from models.encoder import FeatureEncoder
from models.depth_head import DepthHead
from models.gaussian_decoder import GaussianDecoder
from models.pd_block import PoseDrivenFusionBlock
from utils.intrinsics import get_camera_intrinsics
from utils.config import config

# Optional: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
dataset = TUMSequenceDataset(config['dataset_path'], config['sequence_name'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Initialize Models
encoder = FeatureEncoder().to(device)
depth_head = DepthHead().to(device)
gaussian_decoder = GaussianDecoder().to(device)
pd_block = PoseDrivenFusionBlock().to(device)

# Loss and Optimizer
loss_fn = nn.MSELoss()  # Example loss (can be replaced with more appropriate one)
optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(depth_head.parameters()) +
    list(gaussian_decoder.parameters()) +
    list(pd_block.parameters()),
    lr=config['learning_rate']
)

# Training Loop
num_epochs = config['epochs']
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        rgb = batch['rgb'].to(device)                # [B, 3, H, W]
        depth = batch['depth'].to(device)            # [B, 1, H, W]
        pose = batch['pose'].to(device)              # [B, 4, 4]
        intrinsics = get_camera_intrinsics(rgb.shape[-2:], device)  # [B, 3, 3]

        # Forward pass
        features = encoder(rgb)
        depth_pred = depth_head(features)
        world_coords = dataset.backproject(depth_pred, intrinsics)  # [B, 3, H, W]
        fused_feat = pd_block(features, depth_pred, pose, intrinsics)
        scale, color, opacity = gaussian_decoder(fused_feat)

        # Target or pseudo ground-truth can be constructed here
        target = batch['target'].to(device)  # Placeholder for supervision signal
        prediction = torch.cat([scale, color, opacity], dim=1)

        loss = loss_fn(prediction, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save model checkpoint
torch.save({
    'encoder': encoder.state_dict(),
    'depth_head': depth_head.state_dict(),
    'gaussian_decoder': gaussian_decoder.state_dict(),
    'pd_block': pd_block.state_dict()
}, 'checkpoint_drivingrecon.pt')
