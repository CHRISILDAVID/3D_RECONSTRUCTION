"""Test DUSt3R imports to find what's failing."""
import sys
import traceback

sys.path.insert(0, r"E:\Sem 8 Project\backend\dust3r")

print("Step 1: import torch...")
try:
    import torch
    print(f"  OK: torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("Step 2: import dust3r.model...")
try:
    from dust3r.model import AsymmetricCroCo3DStereo
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("Step 3: import dust3r.inference...")
try:
    from dust3r.inference import inference
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("Step 4: import dust3r.utils.image...")
try:
    from dust3r.utils.image import load_images
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("Step 5: import dust3r.image_pairs...")
try:
    from dust3r.image_pairs import make_pairs
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("Step 6: import dust3r.cloud_opt...")
try:
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("\nDone!")
