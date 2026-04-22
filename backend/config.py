"""
Configuration for the 3D Reconstruction Backend
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# DUSt3R model config
DUST3R_MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

# Image processing
MAX_IMAGES = 300
MIN_IMAGES = 2
IMAGE_SIZE = 512  # DUSt3R default input resolution
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Server
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

# Segmentation (Grounding DINO + SAM 2)
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2-hiera-large"
SEGMENTATION_PROMPT = "object . foreground . product . item"
SEGMENTATION_BOX_THRESHOLD = 0.30
SEGMENTATION_TEXT_THRESHOLD = 0.25
