#!/usr/bin/env python3
"""
Download the u2net model for rembg locally.
Run this once to avoid runtime downloads.
"""

import requests
import os
from tqdm import tqdm

print("=== Downloading Rembg u2net Model ===")

# Create models directory
os.makedirs("models", exist_ok=True)

url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
model_path = "models/u2net.onnx"

try:
    print(f"Downloading from: {url}")
    print("This may take 3-5 minutes depending on your connection...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as file, tqdm(
        desc="Downloading u2net.onnx",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n✓ Successfully saved u2net model to {model_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print("  Ready for local loading!")
    
    # Test the download
    print("\n=== Testing model integrity ===")
    from rembg import new_session
    session = new_session(model_name="u2net", file_name=model_path)
    print("✓ Model loads correctly!")
    
except Exception as e:
    print(f"✗ Failed to download u2net model: {e}")
    print("Background removal will use online model (may be slower/unreliable).")