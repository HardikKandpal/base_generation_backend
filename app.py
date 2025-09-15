import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import argparse
from rembg import remove
from PIL import Image
import io
import os
import tempfile
import shutil
from typing import Optional

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse

# --- Initialize FastAPI App ---
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import os
# Global model variable for local loading
model = None
TORCH_AVAILABLE = False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9002"],  # Frontend URL
    allow_credentials=False,
    allow_methods=["POST"],  # Our endpoint method
    allow_headers=["*"],  # Allow all headers (Content-Type for FormData)
)

# --- Local Model Loading ---
import os

def load_local_models():
    global model, TORCH_AVAILABLE
    
    # Check for local Torch model
    torch_model_path = "models/deeplabv3_resnet101_coco.pth"
    if os.path.exists(torch_model_path):
        try:
            print("Loading local DeepLabV3 model...")
            from torchvision.models.segmentation import deeplabv3_resnet101
            import torch
            
            # Load model architecture without weights (no download)
            model = deeplabv3_resnet101(weights=None)
            model.load_state_dict(torch.load(torch_model_path, map_location="cpu"))
            model.eval()
            TORCH_AVAILABLE = True
            print("✓ Local Torch model loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load local Torch model: {e}")
            TORCH_AVAILABLE = False
            model = None
    else:
        print("⚠️ Local Torch model not found, will use online loading if available")
    
    # Check for local Rembg model
    rembg_model_path = "models/u2net.onnx"
    if os.path.exists(rembg_model_path):
        print(f"✓ Local Rembg model found at {rembg_model_path}")

# Load models on startup
load_local_models()


# --- Existing Core Logic (Unchanged) ---
# TORCH_AVAILABLE and model are now set by load_local_models()

TARGET_CLASSES = {
    15: 'person', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 2: 'car', 4: 'motorcycle', 6: 'bus', 8: 'boat'
}

WALL_COUNT_LIMITS = {
    2: 25, 3: 50, 4: 75, 5: 100, 6: 125, 7: 175, 8: 225,
    9: 250, 10: 275, 11: 300, 12: 300, 13: 300, 14: 325,
    15: 325, 16: 325, 17: 325,
}

THEME_PALETTES = {
    # Default theme for when no TH is specified
    'default': {
        'wall_primary': '#0D47A1',   # Dark Blue
        'wall_secondary': '#000000', # Black
    },
    # Themed colors for the grass background
    'grass': {
        'light': '#a5c14a',
        'dark': '#8aab3a',
        'grid_lines': '#799633'
    },
    # --- Town Hall Specific Wall Themes ---
    2: {
        'wall_primary': '#A1887F',   # Light Wood Brown
        'wall_secondary': '#5D4037', # Dark Wood
    },
    3: {
        'wall_primary': '#B0BEC5',   # Pale Stone Gray
        'wall_secondary': '#37474F', # Slate Gray
    },
    4: {
        'wall_primary': '#FFCA28',   # Golden Yellow
        'wall_secondary': '#FFA000', # Deep Amber
    },
    5: {
        'wall_primary': '#FFCA28',   # Golden Yellow
        'wall_secondary': '#FFA000', # Deep Amber
    },
    6: {
        'wall_primary': '#E0E0E0',   # Crystal/Light Gray
        'wall_secondary': '#616161', # Dark Gray
    },
    7: {
        'wall_primary': '#6A1B9A',   # Regal Purple
        'wall_secondary': '#311B92', # Deep Indigo
    },
    8: {
        'wall_primary': '#424242',   # Skull Gray
        'wall_secondary': '#212121', # Obsidian Black
    },
    9: {
        'wall_primary': '#616161',   # Dark Steel Gray
        'wall_secondary': '#212121', # Obsidian Black
    },
    10: {
        'wall_primary': '#F44336',   # Inferno Red
        'wall_secondary': '#B71C1C', # Dark Crimson
    },
    11: {
        'wall_primary': '#FFFFFF',   # Glowing White
        'wall_secondary': '#90A4AE', # Light Steel
    },
    12: {
        'wall_primary': '#00E5FF',   # Bright Electric Blue
        'wall_secondary': '#002768', # Deep Navy Blue
    },
    13: {
        'wall_primary': '#29B6F6',   # Icy Blue
        'wall_secondary': '#01579B', # Deep Cyan
    },
    14: {
        'wall_primary': '#64DD17',   # Vibrant Jungle Green
        'wall_secondary': '#1B5E20', # Dark Forest Green
    },
    15: {
        'wall_primary': '#D500F9',   # Royal Magenta
        'wall_secondary': '#4A148C', # Deep Purple
    },
    16: {
        'wall_primary': '#FFAB00',   # Nature Gold/Orange
        'wall_secondary': '#4E342E', # Earthy Brown
    },
    17: {
        'wall_primary': '#FFD700',   # Mythic Gold
        'wall_secondary': '#212121', # Obsidian Black
    },
}

def skeletonize_image(img):
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_copy = img.copy()
    while True:
        opened = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img_copy, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img_copy = cv2.erode(img_copy, element)
        if cv2.countNonZero(img_copy) == 0: break
    return skeleton

def pixelate_mask(mask, size, scale, tx, ty, content_scale=0.8, threshold_level=100):
    if mask is None or cv2.countNonZero(mask) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    mask = mask[y:y+h, x:x+w]

    new_w = max(1, int(w * content_scale))
    new_h = max(1, int(h * content_scale))
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    centered_mask = np.zeros((h, w), dtype=np.uint8)
    x_offset = (w - new_w) // 2
    y_offset = (h - new_h) // 2
    centered_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_mask

    (h, w) = centered_mask.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -45, scale)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones = np.ones(shape=(len(corners), 1))
    points_ones = np.hstack([corners, ones])
    transformed_corners = M.dot(points_ones.T).T
    min_x, min_y = np.min(transformed_corners, axis=0)
    max_x, max_y = np.max(transformed_corners, axis=0)
    nW = int(np.ceil(max_x - min_x))
    nH = int(np.ceil(max_y - min_y))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated_mask = cv2.warpAffine(centered_mask, M, (nW, nH), borderValue=0)

    pixelated_img = cv2.resize(rotated_mask, (size, size), interpolation=cv2.INTER_AREA)
    M_trans = np.float32([[1, 0, tx], [0, 1, -ty]])
    pixelated_img = cv2.warpAffine(pixelated_img, M_trans, (size, size), borderValue=0)

    _, final_grid_img = cv2.threshold(pixelated_img, threshold_level, 255, cv2.THRESH_BINARY)
    return (final_grid_img > 0).astype(np.uint8)


def get_best_mask(image_path_or_bytes):
    global model, TORCH_AVAILABLE
    
    # Try Torch model if available and loaded
    if TORCH_AVAILABLE and model is not None:
        try:
            print("Attempting high-quality AI object detection (local model)...")
            import torchvision.transforms as T
            import torch
            
            transform = T.Compose([
                T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Handle different input types
            if isinstance(image_path_or_bytes, str):
                input_image = Image.open(image_path_or_bytes).convert("RGB")
            else:
                # For UploadFile, read from file
                if hasattr(image_path_or_bytes, 'seek'):
                    image_path_or_bytes.seek(0)
                input_bytes = image_path_or_bytes.read()
                input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
            
            input_tensor = transform(input_image)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                output = model(input_tensor)['out'][0]
            
            output_predictions = output.argmax(0).byte().cpu().numpy()
            combined_mask = np.zeros_like(output_predictions, dtype=np.uint8)
            
            found_classes = [name for id, name in TARGET_CLASSES.items() if np.any(output_predictions == id)]
            if found_classes:
                print(f"AI model successfully identified: {', '.join(found_classes)}.")
                for id in TARGET_CLASSES.keys():
                    if np.any(output_predictions == id):
                        combined_mask = cv2.bitwise_or(combined_mask, (output_predictions == id).astype(np.uint8))
                mask_cv = (combined_mask * 255)
                return cv2.resize(mask_cv, (input_image.width, input_image.height), interpolation=cv2.INTER_NEAREST)
            else:
                print("AI model did not find any of the target subjects.")
                
        except Exception as e:
            print(f"Error during AI Semantic analysis (local model): {e}")
            # Continue to background removal fallback
    
    # Background removal fallback
    print("\n--- Falling back to general background removal. ---")
    try:
        # Handle file pointer reset for UploadFile
        if hasattr(image_path_or_bytes, 'seek'):
            image_path_or_bytes.seek(0)
        
        input_bytes = image_path_or_bytes.read() if hasattr(image_path_or_bytes, 'read') else open(image_path_or_bytes, 'rb').read()
        
        
        
        # Background removal - try local first, fallback to default
        print("Performing background removal...")
        try:
            from rembg import remove
            # Default rembg will automatically use local model if available
            # The u2net.onnx in models/ will be detected by rembg
            output_bytes = remove(input_bytes)
            pil_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
            img_with_alpha = np.array(pil_img)
            alpha_channel = img_with_alpha[:, :, 3]
            print(f"✓ Background removal successful (shape: {alpha_channel.shape})")
            return alpha_channel
        except Exception as rembg_error:
            print(f"Rembg failed, using OpenCV edge detection: {rembg_error}")
        
    except Exception as e:
        print(f"All background removal methods failed, using OpenCV edge detection: {e}")
        # Final fallback: OpenCV edge detection
        try:
            if isinstance(image_path_or_bytes, str):
                input_img = cv2.imread(image_path_or_bytes)
            else:
                # For bytes/UploadFile
                if hasattr(image_path_or_bytes, 'seek'):
                    image_path_or_bytes.seek(0)
                input_bytes = image_path_or_bytes.read()
                input_img = cv2.imdecode(np.frombuffer(input_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if input_img is None:
                print("Could not decode image for OpenCV fallback")
                return None
                
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            return edges
            
        except Exception as cv2_error:
            print(f"OpenCV fallback also failed: {cv2_error}")
            return None

def generate_internal_details(image_path, base_mask, args):
    if base_mask is None: return None
    h, w = base_mask.shape
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (w, h))

    kernel = np.ones((5,5), np.uint8)
    inner_mask = cv2.erode(base_mask, kernel, iterations=1)

    isolated_subject = cv2.bitwise_and(original_img, original_img, mask=inner_mask)
    filtered_img = cv2.bilateralFilter(isolated_subject, args.bilateral_d, args.sigma_color, args.sigma_space)
    pixels = np.float32(filtered_img.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, args.k_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception as e:
        print(f"[ERROR] K-means failed: {e}")
        return None
    if len(centers) == 0 or np.all(centers == 0): return None

    centers = np.uint8(centers)
    quantized_img = centers[labels.flatten()].reshape(filtered_img.shape)

    detail_mask = np.zeros_like(base_mask)
    denoise_kernel_size = args.denoise
    if denoise_kernel_size > 0:
        denoise_kernel = np.ones((denoise_kernel_size, denoise_kernel_size), np.uint8)

    for color_center in centers:
        if np.all(color_center == [0, 0, 0]): continue
        color_mask = cv2.inRange(quantized_img, color_center, color_center)
        if denoise_kernel_size > 0:
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, denoise_kernel)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > args.min_contour_area:
                cv2.drawContours(detail_mask, [contour], -1, (255), thickness=args.contour_thickness)
    if np.count_nonzero(detail_mask) == 0: return None
    return detail_mask

def visualize_rotated_diamond_grid(grid, output_path):
    """
    Visualizes the grid with a themed, checkerboard grass background and
    a consistent, high-contrast black wall theme.
    """
    wall_count = np.sum(grid)
    if grid is None:
        print("No grid to display.")
        return

    # --- THEME SELECTION ---
    # We only need the grass palette now. Wall colors will be hardcoded.
    grass_palette = THEME_PALETTES['grass']

    size = grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor(grass_palette['dark']) # Fallback color

    grid_center = np.array([size/2, size/2])
    angle_rad = np.deg2rad(45)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    half_size = 0.5 / np.sqrt(2)
    square_corners = np.array([[-half_size, -half_size], [half_size, -half_size], [half_size, half_size], [-half_size, half_size]])

    for i in range(size):
        for j in range(size):
            center = np.array([j, i])
            rotated_center = R @ (center - grid_center) + grid_center
            polygon_verts = square_corners @ R.T + rotated_center

            # --- 1. DRAW THE GRASS BACKGROUND (CHECKERBOARD) ---
            bg_color = grass_palette['light'] if (i + j) % 2 == 0 else grass_palette['dark']
            background_patch = Polygon(
                polygon_verts,
                closed=True,
                facecolor=bg_color,
                edgecolor=grass_palette['grid_lines'],
                linewidth=0.7
            )
            ax.add_patch(background_patch)

            # --- 2. DRAW THE WALLS (ON TOP OF THE GRASS) ---
            if grid[i, j] == 1:
                wall_patch = Polygon(
                    polygon_verts,
                    closed=True,
                    # --- MODIFICATION: Hardcoded black theme for all walls ---
                    facecolor='#000000',      # Primary wall color is now black
                    edgecolor='#212121',    # Edge color is a very dark gray for a subtle 3D effect
                    linewidth=1.2
                )
                ax.add_patch(wall_patch)

    all_corners = np.array([[0, 0], [size, 0], [0, size], [size, size]]) - grid_center
    rotated_corners = all_corners @ R.T + grid_center
    min_x, min_y = rotated_corners.min(axis=0) - 1
    max_x, max_y = rotated_corners.max(axis=0) + 1
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f'Walls: {int(wall_count)}', color='white', y=0.98)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200, facecolor=ax.get_facecolor())
    plt.close(fig)
    print(f"Themed image successfully saved to {output_path}")

def generate_and_limit_grid(final_mask, args, wall_limit, mode):
    def _generate(scale):
        pixelated = pixelate_mask(
            final_mask, args.grid_size, args.scale, args.x_offset, args.y_offset,
            content_scale=scale, threshold_level=args.threshold_level
        )
        # --- MODIFIED: Only skeletonize 'outline' mode ---
        if mode == 'outline':
            grid = skeletonize_image(pixelated * 255)
            return (grid > 0).astype(np.uint8)
        # For 'portrait' and 'silhouette', return the solid shape
        return (pixelated > 0).astype(np.uint8)

    current_scale = args.content_scale
    final_grid = _generate(current_scale)
    current_count = np.sum(final_grid)

    if current_count <= wall_limit:
        print(f"Initial wall count ({current_count}) is within the limit ({wall_limit}). No scaling needed.")
        return final_grid
    
    print(f"Initial wall count ({current_count}) exceeds limit ({wall_limit}). Starting dynamic scaling...")

    max_iterations = 10
    for i in range(max_iterations):
        ratio = wall_limit / current_count
        adjustment_factor = np.sqrt(ratio)
        current_scale *= (adjustment_factor * 0.9 + 0.1) 
        
        final_grid = _generate(current_scale)
        last_count = current_count
        current_count = np.sum(final_grid)

        print(f"  Attempt {i+1}: Scale adjusted to {current_scale:.3f} -> New wall count: {current_count}")
        
        if current_count <= wall_limit:
            print(f"Success! Final wall count ({current_count}) is within the limit.")
            return final_grid
        if current_count >= last_count:
            print("[WARN] Scaling is not reducing wall count effectively. Stopping.")
            return final_grid

    print(f"[WARN] Could not meet wall limit after {max_iterations} attempts. Returning best effort with {current_count} walls.")
    return final_grid


# --- FastAPI Endpoint ---
@app.post("/generate-layout/")
async def create_layout(
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(...),
    mode: str = Form('portrait'),
    th: Optional[str] = Form(None),
    detail: int = Form(10),
    denoise: int = Form(5),
    noise_filter: int = Form(50),
    thickness: int = Form(4),
    scale: float = Form(1.0),
    x_offset: int = Form(0),
    y_offset: int = Form(0),
    grid_size: int = Form(44),
    content_scale: float = Form(0.8),
    threshold_level: int = Form(100),
    bilateral_d: int = Form(9),
    sigma_color: int = Form(75),
    sigma_space: int = Form(75)
):
    # Create a temporary directory to store files
    temp_dir = tempfile.mkdtemp()
    # Define paths for temporary input and output files
    input_path = os.path.join(temp_dir, image_file.filename)
    output_path = os.path.join(temp_dir, "output.png")

    # Save the uploaded file to the temporary input path
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    
    # After the response is sent, clean up the temporary directory
    background_tasks.add_task(shutil.rmtree, temp_dir)

    # --- Replicate argparse behavior for compatibility with existing functions ---
    args = argparse.Namespace()
    args.mode = mode
    args.th = th
    args.detail = detail
    args.k_colors = detail
    args.denoise = denoise
    args.noise_filter = noise_filter
    args.min_contour_area = noise_filter
    args.thickness = thickness
    args.contour_thickness = max(1, thickness - 2)
    args.scale = scale
    args.x_offset = x_offset
    args.y_offset = y_offset
    args.grid_size = grid_size
    args.content_scale = content_scale
    args.threshold_level = threshold_level
    args.bilateral_d = bilateral_d
    args.sigma_color = sigma_color
    args.sigma_space = sigma_space
    
    # --- Start of the original main() logic ---
    wall_limit = None
    if args.th:
        try:
            th_level_str = ''.join(filter(str.isdigit, args.th))
            if th_level_str:
                th_level = int(th_level_str)
                if th_level in WALL_COUNT_LIMITS:
                    wall_limit = WALL_COUNT_LIMITS[th_level]
                    print(f"--- Enforcing wall count limit for TH{th_level}: {wall_limit} walls ---")
            else: raise ValueError("No digits")
        except (ValueError, TypeError):
            print(f"[WARN] Invalid TH format: '{args.th}'")

    print("--- Starting Layout Generation ---")
    base_mask = get_best_mask(input_path)
    if base_mask is None: return {"error": "Could not create a mask from the subject."}

    final_mask = None
    if args.mode == 'portrait':
        print("Generating detailed portrait...")
        args.k_colors = max(args.detail, 40)
        args.min_contour_area = max(10, args.noise_filter // 5)
        args.denoise = 0
        detail_mask = generate_internal_details(input_path, base_mask, args)
        contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_mask = np.zeros_like(base_mask)
        cv2.drawContours(outline_mask, contours, -1, (255), thickness=args.thickness)
        if detail_mask is not None and np.count_nonzero(detail_mask) > 0:
            final_mask = cv2.bitwise_or(outline_mask, detail_mask)
        else:
            final_mask = outline_mask
    elif args.mode == 'outline':
        print("Generating clean external outline...")
        contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(base_mask)
        cv2.drawContours(final_mask, contours, -1, (255), thickness=args.thickness)
    elif args.mode == 'silhouette':
        print("Generating simple silhouette...")
        final_mask = base_mask

    if final_mask is not None:
        if wall_limit is not None:
            final_grid = generate_and_limit_grid(final_mask, args, wall_limit, args.mode)
        else:
            pixelated_shape = pixelate_mask(
                final_mask, args.grid_size, args.scale, args.x_offset, args.y_offset,
                content_scale=args.content_scale, threshold_level=args.threshold_level
            )
            # --- MODIFIED: Only skeletonize 'outline' mode ---
            if args.mode == 'outline':
                final_grid = skeletonize_image(pixelated_shape * 255)
                final_grid = (final_grid > 0).astype(np.uint8)
            else: # For 'portrait' and 'silhouette', use the solid shape
                final_grid = (pixelated_shape > 0).astype(np.uint8)

        flipped_grid = np.flipud(final_grid)
        visualize_rotated_diamond_grid(flipped_grid, output_path=output_path)

        # Return the generated image file
        return FileResponse(output_path, media_type="image/png")

    return {"error": "Could not generate a final mask."}