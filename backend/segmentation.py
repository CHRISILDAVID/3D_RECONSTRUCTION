"""
Segmentation Pre-Processing for 3D Reconstruction

Uses Grounding DINO (tiny) for automatic object detection and
SAM 2 (large) for pixel-accurate mask generation.

Models are loaded, used, then fully unloaded from VRAM before
DUSt3R loads its own weights — sequential strategy for GTX 1650 (4GB).
"""

import gc
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Lazy imports (avoid loading at module level) ──────────────────────────────
SEGMENTATION_AVAILABLE = False
_import_error = None

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    _import_error = "PyTorch not available"

if _TORCH_OK:
    try:
        import transformers  # Grounding DINO lives in HF transformers
        SEGMENTATION_AVAILABLE = True
        logger.info("transformers available for Grounding DINO")
    except ImportError as e:
        _import_error = f"transformers not installed: {e}"
        logger.warning(_import_error)


class SegmentationPipeline:
    """
    Automatic foreground segmentation using Grounding DINO + SAM 2.

    Workflow per image:
      1. Grounding DINO detects bounding boxes for the main object.
      2. The best box (largest & most central) is selected.
      3. SAM 2 refines the box into a pixel-accurate mask.
      4. The mask is applied; background pixels are set to black.
    """

    # Model IDs
    DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    SAM2_MODEL_ID = "facebook/sam2-hiera-large"

    # Detection parameters
    BOX_THRESHOLD = 0.30
    TEXT_THRESHOLD = 0.25
    # Generic prompt — no user input needed; detects any salient foreground object
    DETECTION_PROMPT = "object . foreground . product . item"

    def __init__(self, device: str = "cuda"):
        self.device = device if (
            _TORCH_OK and torch.cuda.is_available() and device == "cuda"
        ) else "cpu"
        self._dino_processor = None
        self._dino_model = None
        self._sam2_predictor = None

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load_models(self, progress_callback: Optional[Callable] = None):
        """Load Grounding DINO and SAM 2 onto device."""
        cb = progress_callback or (lambda *a: None)

        if not SEGMENTATION_AVAILABLE:
            logger.warning(f"Segmentation unavailable: {_import_error}")
            cb("segmenting", 10, f"Segmentation skipped: {_import_error}")
            return

        # ── Grounding DINO ──
        cb("segmenting", 5, "Loading Grounding DINO (object detector)...")
        logger.info(f"Loading Grounding DINO from {self.DINO_MODEL_ID}")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._dino_processor = AutoProcessor.from_pretrained(self.DINO_MODEL_ID)
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.DINO_MODEL_ID
            ).to(self.device)
            self._dino_model.eval()
            logger.info("Grounding DINO loaded")
            cb("segmenting", 8, "Grounding DINO ready")
        except Exception as e:
            logger.error(f"Grounding DINO load failed: {e}", exc_info=True)
            self._dino_model = None

        # ── SAM 2 ──
        cb("segmenting", 9, "Loading SAM 2 large (mask predictor)...")
        logger.info(f"Loading SAM 2 from {self.SAM2_MODEL_ID}")
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
                self.SAM2_MODEL_ID,
                device=self.device,
            )
            logger.info("SAM 2 large loaded")
            cb("segmenting", 11, "SAM 2 ready")
        except Exception as e:
            logger.error(f"SAM 2 load failed: {e}", exc_info=True)
            self._sam2_predictor = None

    def unload_models(self):
        """Delete model weights and free VRAM for DUSt3R."""
        logger.info("Unloading segmentation models to free VRAM...")
        del self._dino_model, self._dino_processor, self._sam2_predictor
        self._dino_model = None
        self._dino_processor = None
        self._sam2_predictor = None
        if _TORCH_OK:
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Segmentation models unloaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def segment_images(
        self,
        image_paths: List[str],
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        Segment all images and save masked versions to output_dir.

        Returns a list of paths to the masked images. If segmentation
        fails for any image, the original image path is returned for
        that image (graceful fallback, reconstruction still proceeds).
        """
        cb = progress_callback or (lambda *a: None)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._models_available():
            logger.warning("No segmentation models loaded — returning original paths")
            return image_paths

        masked_paths = []
        n = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            pct = 11 + int(4 * (idx + 1) / n)
            cb("segmenting", pct, f"Segmenting image {idx+1}/{n}...")

            try:
                masked_path = self._process_one_image(img_path, output_dir, idx)
                masked_paths.append(masked_path)
                logger.info(f"  [{idx+1}/{n}] Masked: {masked_path}")
            except Exception as e:
                logger.warning(f"  [{idx+1}/{n}] Segmentation failed for {img_path}: {e}")
                # Fallback: copy original to output_dir under a consistent name
                fallback = output_dir / f"masked_{idx:04d}{Path(img_path).suffix}"
                shutil.copy2(img_path, fallback)
                masked_paths.append(str(fallback))

        logger.info(f"Segmentation complete: {len(masked_paths)} images processed")
        return masked_paths

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _models_available(self) -> bool:
        return self._dino_model is not None or self._sam2_predictor is not None

    def _process_one_image(self, img_path: str, output_dir: Path, idx: int) -> str:
        """Run DINO + SAM 2 on one image and save the masked result."""
        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)

        # Step 1: Grounding DINO → bounding boxes
        bbox = self._detect_main_object(pil_img)

        if bbox is None:
            logger.info(f"    No object detected in {Path(img_path).name} — using original")
            out_path = output_dir / f"masked_{idx:04d}{Path(img_path).suffix}"
            shutil.copy2(img_path, out_path)
            return str(out_path)

        # Step 2: SAM 2 → pixel mask
        mask = self._get_sam_mask(np_img, bbox)

        if mask is None:
            logger.info(f"    SAM 2 mask failed for {Path(img_path).name} — using original")
            out_path = output_dir / f"masked_{idx:04d}{Path(img_path).suffix}"
            shutil.copy2(img_path, out_path)
            return str(out_path)

        # Step 3: Apply mask — background → black
        masked_np = np_img.copy()
        masked_np[~mask] = 0  # zero out background

        # Step 4: Save
        out_path = output_dir / f"masked_{idx:04d}{Path(img_path).suffix}"
        masked_pil = Image.fromarray(masked_np.astype(np.uint8))
        masked_pil.save(str(out_path))

        fg_pct = 100 * mask.sum() / mask.size
        logger.info(f"    Mask applied: {fg_pct:.1f}% foreground retained")
        return str(out_path)

    def _detect_main_object(self, pil_img: Image.Image) -> Optional[List[float]]:
        """
        Run Grounding DINO and return the best bounding box [x0, y0, x1, y1]
        in pixel coordinates, or None if nothing is detected confidently.

        Selection strategy:
          - Filter by box confidence > BOX_THRESHOLD
          - Among remaining boxes, pick the one that scores highest on a
            combined metric: normalised_area * centrality
          - This favours large, roughly-centred subjects over small edge objects
        """
        if self._dino_model is None:
            return None

        W, H = pil_img.size

        inputs = self._dino_processor(
            images=pil_img,
            text=self.DETECTION_PROMPT,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._dino_model(**inputs)

        # Post-process — newer transformers removed box_threshold /
        # text_threshold kwargs, so we always call without them and
        # filter manually afterwards.
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[(H, W)],
        )[0]

        boxes = results["boxes"].cpu().numpy()   # shape (N, 4) xyxy pixels
        scores = results["scores"].cpu().numpy()  # shape (N,)

        # Apply confidence threshold manually
        keep = scores >= self.BOX_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]

        if len(boxes) == 0:
            return None

        best_box = self._select_best_box(boxes, scores, W, H)
        return best_box.tolist()

    def _select_best_box(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        W: int,
        H: int,
    ) -> np.ndarray:
        """Score each box by confidence × area × centrality; return the best."""
        img_area = W * H
        cx_img, cy_img = W / 2.0, H / 2.0

        combined = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            norm_area = area / img_area

            # Centrality: 1 if centre of box == centre of image, 0 at edge
            cx_box = (x0 + x1) / 2.0
            cy_box = (y0 + y1) / 2.0
            dist = np.sqrt(((cx_box - cx_img) / W) ** 2 + ((cy_box - cy_img) / H) ** 2)
            centrality = max(0.0, 1.0 - dist * 2.0)  # linear decay, clipped at 0

            # Combine: weight area 60%, centrality 40%
            composite = score * (0.6 * norm_area + 0.4 * centrality)
            combined.append(composite)

        best_idx = int(np.argmax(combined))
        return boxes[best_idx]

    def _get_sam_mask(
        self,
        np_img: np.ndarray,
        bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Feed bounding box to SAM 2 and return a boolean mask (H, W).
        Uses the highest-scoring mask among SAM 2's candidates.
        """
        if self._sam2_predictor is None:
            # Fallback: construct a rough rectangular mask from the DINO bbox
            return self._bbox_to_mask(np_img.shape[:2], bbox)

        with torch.inference_mode():
            self._sam2_predictor.set_image(np_img)
            box_arr = np.array(bbox, dtype=np.float32)
            masks, iou_scores, _ = self._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_arr[None, :],   # SAM 2 expects (1, 4)
                multimask_output=True,  # get multiple candidates
            )

        # masks shape: (N, H, W) — pick highest IOU
        best = int(np.argmax(iou_scores))
        mask = masks[best].astype(bool)
        return mask

    def _bbox_to_mask(
        self,
        shape: Tuple[int, int],
        bbox: List[float],
    ) -> np.ndarray:
        """Coarse rectangular mask fallback when SAM 2 is unavailable."""
        H, W = shape
        mask = np.zeros((H, W), dtype=bool)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        mask[y0:y1, x0:x1] = True
        return mask
