# file: run_viewer.py
import torch
from visualize_gaussians_panda import visualize_gaussians_panda

# Load saved tensor from previous step
gaussians = torch.load("driving_scene.pt")  # or replace with your .pt/.pth file
visualize_gaussians_panda(gaussians)
