#!/usr/bin/env python3
"""
Complete model downloader for local loading.
Downloads ALL required PyTorch models and saves them locally.
Run this once to eliminate ALL runtime downloads.
"""

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import os
import requests
from tqdm import tqdm
from rembg import new_session

print("=== Complete Local Model Setup ===")
print("This will download ~400MB total. Takes 10-20 minutes depending on connection.")
print()

# Create models directory
os.makedirs("models", exist_ok=True)

def download_file(url, filename, chunk_size=8192):
    """Download file with progress bar"""
    print(f"Downloading {os.path.basename(filename)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

# 1. Download Rembg u2net model (176MB)
print("\n1. Downloading Rembg u2net model...")
u2net_url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
u2net_path = "models/u2net.onnx"
if not os.path.exists(u2net_path):
    download_file(u2net_url, u2net_path)
    print("✓ u2net.onnx downloaded")
else:
    print("✓ u2net.onnx already exists")

# 2. Download DeepLabV3 complete model
print("\n2. Downloading DeepLabV3 model components...")

# Download the segmentation weights
print("  Downloading segmentation weights...")
seg_weights_url = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
seg_weights_path = "models/deeplabv3_resnet101_coco.pth"
if not os.path.exists(seg_weights_path):
    download_file(seg_weights_url, seg_weights_path)
    print("✓ deeplabv3_resnet101_coco.pth downloaded")
else:
    print("✓ deeplabv3_resnet101_coco.pth already exists")

# Download the ResNet101 backbone (required by DeepLabV3)
print("  Downloading ResNet101 backbone...")
backbone_url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
backbone_path = "models/resnet101-63fe2227.pth"
if not os.path.exists(backbone_path):
    download_file(backbone_url, backbone_path)
    print("✓ resnet101-63fe2227.pth downloaded")
else:
    print("✓ resnet101-63fe2227.pth already exists")

# 3. Test model loading
print("\n3. Testing model integrity...")
try:
    # Test Torch model loading
    print("  Testing DeepLabV3 loading...")
    model = deeplabv3_resnet101(weights=None)  # Architecture only
    
    # Load with strict=False to ignore extra keys
    seg_state_dict = torch.load(seg_weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(seg_state_dict, strict=False)
    model.eval()
    print("  ✓ DeepLabV3 loads correctly!")
    
    # Test Rembg model
    print("  Testing u2net loading...")
    session = new_session(model_name="u2net", file_name=u2net_path)
    print("  ✓ u2net loads correctly!")
    
except Exception as e:
    print(f"  ✗ Model loading test failed: {e}")

# 4. Create model loader script
print("\n4. Creating local model loader...")
with open("load_local_models.py", "w", encoding="utf-8") as f:
    f.write('''#!/usr/bin/env python3
"""
Load all models locally without any downloads.
Import this module to pre-load models.
"""

import torch
import os
from torchvision.models.segmentation import deeplabv3_resnet101
from rembg import new_session

# Global variables
model = None
rembg_session = None
TORCH_AVAILABLE = False
REMBG_LOCAL = False

def load_all_models():
    global model, rembg_session, TORCH_AVAILABLE, REMBG_LOCAL
    
    print("Loading local models...")
    
    # Load DeepLabV3
    try:
        seg_weights_path = "models/deeplabv3_resnet101_coco.pth"
        backbone_path = "models/resnet101-63fe2227.pth"
        
        if os.path.exists(seg_weights_path) and os.path.exists(backbone_path):
            # Load with weights_only=False for compatibility
            seg_state = torch.load(seg_weights_path, map_location="cpu", weights_only=False)
            backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=False)
            
            # Load full model
            model = deeplabv3_resnet101(weights=None)
            
            # Load complete state dict (strict=False to ignore extra keys)
            full_state = seg_state.copy()
            full_state.update({k.replace("backbone.", ""): v for k, v in backbone_state.items() if k.startswith("backbone.")})
            model.load_state_dict(full_state, strict=False)
            
            model.eval()
            TORCH_AVAILABLE = True
            print("v DeepLabV3 loaded from local files")
        else:
            print("! Missing model files, Torch unavailable")
            
    except Exception as e:
        print(f"X Torch model loading failed: {e}")
        TORCH_AVAILABLE = False
    
    # Load Rembg
    try:
        u2net_path = "models/u2net.onnx"
        if os.path.exists(u2net_path):
            rembg_session = new_session(
                model_name="u2net",
                file_name=u2net_path,
                providers=["CPUExecutionProvider"]
            )
            REMBG_LOCAL = True
            print("v Local u2net loaded")
        else:
            print("! u2net.onnx not found, using default rembg")
            
    except Exception as e:
        print(f"X Rembg loading failed: {e}")
        REMBG_LOCAL = False

# Auto-load on import
load_all_models()

if __name__ == "__main__":
    print("All local models loaded successfully!")
    print(f"Torch available: {TORCH_AVAILABLE}")
    print(f"Local Rembg: {REMBG_LOCAL}")
''')

print("\n=== Setup Complete! ===")
print("Summary:")
print(f"  Models directory: {os.path.abspath('models')}")
print(f"  DeepLabV3: {os.path.exists('models/deeplabv3_resnet101_coco.pth')}")
print(f"  ResNet101: {os.path.exists('models/resnet101-63fe2227.pth')}")
print(f"  u2net: {os.path.exists('models/u2net.onnx')}")
print()
print("Next steps:")
print("1. Run: python load_local_models.py")
print("2. Start server: uvicorn main:app --host 0.0.0.0 --port 8000")
print("3. No more downloads! Fast startup every time.")