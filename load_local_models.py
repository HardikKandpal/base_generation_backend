#!/usr/bin/env python3
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
