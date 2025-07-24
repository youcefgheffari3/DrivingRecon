from loader.tum_loader import load_tum_sequence

frames = load_tum_sequence("data/fr1_desk", max_frames=50)

print(f"Loaded {len(frames)} frames")

for i, frame in enumerate(frames):
    rgb = frame['rgb']       # [3, H, W]
    depth = frame['depth']   # [1, H, W]
    R = frame['R']           # [3, 3]
    T = frame['T']           # [3]
