import torch
from models.image_encoder import ImageEncoder
import torchvision.transforms as T
from PIL import Image
import os

# ✅ Your correct image path
image_path = r"C:\Users\Gheffari Youcef\Videos\DrivingRecon\data\Driving_in_Finland.jpg"

# Try to load image
if not os.path.exists(image_path):
    print(f"❌ File does not exist: {image_path}")
    exit(1)

image = Image.open(image_path).convert('RGB')

# Preprocessing
transform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(image).unsqueeze(0)

# Run encoder
encoder = ImageEncoder()
encoder.eval()
features = encoder(img_tensor)

print("✅ Feature map shape:", features.shape)
