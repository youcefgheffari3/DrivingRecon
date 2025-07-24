# tum_loader.py

import os
import numpy as np
import torch
import cv2
from PIL import Image

def read_associations(path):
    assoc_path = os.path.join(path, 'associations.txt')
    associations = []
    with open(assoc_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            associations.append((parts[0], parts[1], parts[2], parts[3]))
    return associations

def load_groundtruth(path):
    pose_dict = {}
    with open(os.path.join(path, 'groundtruth.txt')) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            pose_dict[str(t)] = (tx, ty, tz, qx, qy, qz, qw)
    return pose_dict

def quat_to_matrix(q):
    """ Convert quaternion to rotation matrix """
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float32)
    return R

def load_tum_sequence(data_path, target_size=(256, 512), max_frames=None):
    associations = read_associations(data_path)
    gt_poses = load_groundtruth(data_path)

    frames = []

    for i, (t_rgb, rgb_file, t_depth, depth_file) in enumerate(associations):
        if max_frames and i >= max_frames:
            break

        rgb_path = os.path.join(data_path, rgb_file)
        depth_path = os.path.join(data_path, depth_file)

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        rgb = cv2.resize(rgb, target_size)
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float() / 5000.0  # [1,H,W], in meters

        if t_rgb not in gt_poses:
            continue

        tx, ty, tz, qx, qy, qz, qw = gt_poses[t_rgb]
        R = torch.from_numpy(quat_to_matrix([qx, qy, qz, qw]))
        T = torch.tensor([tx, ty, tz], dtype=torch.float32)

        frames.append({
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "R": R,
            "T": T,
            "time": float(t_rgb)
        })

    return frames
