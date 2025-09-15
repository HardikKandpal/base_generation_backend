#!/usr/bin/env python3
"""
Download and cache the DeepLabV3 Torch model locally.
Run this once to avoid runtime downloads.
"""

import torch
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import os

print("=== Downloading DeepLabV3 Model ===")

# Create models directory
os.makedirs("models", exist_ok=True)

try:
    print("Loading model with DEFAULT weights (this triggers download)...")
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).eval()
    
    # Save the model state dict locally
    model_path = "models/deeplabv3_resnet101_coco.pth"
    torch.save(model.state_dict(), model_path)
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Successfully saved DeepLabV3 model to {model_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print("  Ready for local loading!")
    
except Exception as e:
    print(f"✗ Failed to download/save Torch model: {e}")
    print("You can still use OpenCV fallback, but AI detection won't work.")