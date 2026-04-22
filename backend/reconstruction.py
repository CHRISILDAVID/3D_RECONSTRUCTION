"""
3D Reconstruction Pipeline — DUSt3R Dense Reconstruction

Uses DUSt3R (naver/dust3r) for dense 3D point cloud generation.
Designed for GTX 1650 (4GB VRAM): uses 512px native resolution + fp16.
Falls back to OpenCV SfM if DUSt3R is unavailable.
"""
import json
import logging
import struct
import sys
import os
import gc
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List, Set

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Add dust3r to Python path ────────────────────────────────────────────────
DUST3R_DIR = str(Path(__file__).resolve().parent / "dust3r")
# Always force it to position 0 so it's found before any stale entries
if DUST3R_DIR in sys.path:
    sys.path.remove(DUST3R_DIR)
sys.path.insert(0, DUST3R_DIR)
logger.info(f"dust3r path: {DUST3R_DIR}")

# ── Try to import DUSt3R ─────────────────────────────────────────────────────
DUST3R_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

if TORCH_AVAILABLE:
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        DUST3R_AVAILABLE = True
        logger.info("DUSt3R modules imported successfully")
    except ImportError as e:
        logger.warning(f"DUSt3R import failed: {e}")
    except Exception as e:
        logger.warning(f"DUSt3R import error: {e}")


class ReconstructionPipeline:
    """Pipeline that uses DUSt3R for dense 3D reconstruction."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        # 512 = DUSt3R native resolution (trained on this)
        # 224 would lose too much detail and break matching
        self.image_size = 512

    def load_model(self, progress_callback=None):
        """Load the DUSt3R model from HuggingFace."""
        cb = progress_callback or (lambda *a: None)

        if not DUST3R_AVAILABLE:
            logger.warning("DUSt3R not available — will use OpenCV SfM fallback")
            cb("model_loaded", 10, "DUSt3R not available — using OpenCV SfM fallback")
            return

        if self.model is not None:
            logger.info("Model already loaded")
            cb("model_loaded", 10, "Model already loaded")
            return

        try:
            cb("loading_model", 5, "Downloading DUSt3R model (~400MB first time)...")
            logger.info("Loading DUSt3R model from HuggingFace...")
            model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
            logger.info("Model downloaded/loaded from cache")

            # Determine device
            if self.device == "cuda" and torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f} GB")

                self.model = self.model.to(self.device)
                logger.info(f"DUSt3R model moved to {self.device}")
            else:
                self.device = "cpu"
                self.model = self.model.to("cpu")
                logger.info("DUSt3R model on CPU (no CUDA)")

            self.model.eval()
            cb("model_loaded", 10, f"DUSt3R ready on {self.device}")
            logger.info(f"✓ DUSt3R model ready on {self.device}")

        except Exception as e:
            logger.error(f"FAILED to load DUSt3R model: {e}", exc_info=True)
            self.model = None
            cb("model_loaded", 10, f"DUSt3R load failed: {e}")

    def reconstruct(self, image_dir, output_dir, progress_callback=None, segment: bool = False):
        """Run reconstruction. Uses DUSt3R if available, else OpenCV SfM."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        use_dust3r = DUST3R_AVAILABLE and self.model is not None
        logger.info(f"Reconstruction: DUST3R_AVAILABLE={DUST3R_AVAILABLE}, "
                    f"model_loaded={self.model is not None}, using={'DUSt3R' if use_dust3r else 'SfM'}, "
                    f"segment={segment}")

        if use_dust3r:
            return self._dust3r_reconstruct(image_dir, output_dir, progress_callback, segment=segment)
        else:
            logger.warning("Falling back to OpenCV SfM (DUSt3R not available)")
            return self._sfm_reconstruct(image_dir, output_dir, progress_callback)

    # =========================================================================
    # DUSt3R Dense Reconstruction
    # =========================================================================

    def _dust3r_reconstruct(self, image_dir: Path, output_dir: Path, progress_callback, segment: bool = False) -> dict:
        """Dense 3D reconstruction using DUSt3R @ 512px with fp16."""
        cb = progress_callback or (lambda *a: None)
        image_dir = Path(image_dir)

        # --- 1. Discover images ---
        cb("loading_images", 10, "Discovering images...")
        image_paths = sorted([
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        ])
        n_total = len(image_paths)
        logger.info(f"Found {n_total} images in {image_dir}")

        if n_total < 2:
            raise ValueError("Need at least 2 images")

        # --- 1b. Optional segmentation pre-processing ---
        if segment:
            logger.info("Segmentation enabled — masking foreground objects...")
            cb("segmenting", 11, "Loading segmentation models (DINO + SAM 2)...")
            try:
                from segmentation import SegmentationPipeline
                seg_pipeline = SegmentationPipeline(device=self.device)
                seg_pipeline.load_models(progress_callback=cb)
                masked_dir = output_dir / "masked_inputs"
                image_paths = seg_pipeline.segment_images(
                    image_paths, masked_dir, progress_callback=cb
                )
                seg_pipeline.unload_models()
                logger.info(f"Segmentation done — {len(image_paths)} masked images ready")
                cb("segmenting", 15, f"Segmentation complete: {len(image_paths)} images masked")
            except Exception as e:
                logger.warning(f"Segmentation failed — proceeding without masking: {e}")
                cb("segmenting", 15, f"Segmentation skipped (error): {e}")

        # --- 2. Load images ---
        # For 4GB VRAM at 512px: limit to 12 images
        # DUSt3R global alignment VRAM scales with n^2
        max_images = 12
        if len(image_paths) > max_images:
            indices = np.linspace(0, len(image_paths) - 1, max_images, dtype=int)
            image_paths = [image_paths[i] for i in indices]
            logger.info(f"Sampled {max_images} of {n_total} images (VRAM limit)")

        n = len(image_paths)
        cb("loading_images", 16, f"Loading {n} images at {self.image_size}px...")

        images = load_images(image_paths, size=self.image_size)
        logger.info(f"Loaded {n} images at {self.image_size}px")
        cb("loading_images", 19, f"Loaded {n} images")

        # --- 2. Create image pairs ---
        cb("matching", 20, "Creating image pairs...")

        # Use dense pair graphs for better coverage
        if n <= 5:
            scene_graph = "complete"
        elif n <= 8:
            scene_graph = "swin-4"
        else:
            scene_graph = "swin-3"

        pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        n_pairs = len(pairs)
        logger.info(f"Created {n_pairs} image pairs ({scene_graph})")
        cb("matching", 22, f"{n_pairs} pairs to process")

        # --- 3. DUSt3R pairwise inference ---
        cb("matching", 25, f"Running DUSt3R stereo on {n_pairs} pairs...")
        logger.info(f"Starting DUSt3R inference: {n_pairs} pairs, device={self.device}")

        # Clear VRAM before heavy computation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"VRAM before inference: "
                       f"{torch.cuda.memory_allocated()/1024**2:.0f}MB allocated, "
                       f"{torch.cuda.memory_reserved()/1024**2:.0f}MB reserved")

        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    output = inference(pairs, self.model, self.device, batch_size=1)
            else:
                output = inference(pairs, self.model, self.device, batch_size=1)

        logger.info("DUSt3R inference complete")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"VRAM after inference: "
                       f"{torch.cuda.memory_allocated()/1024**2:.0f}MB")

        cb("matching", 50, "Pairwise stereo complete")

        # --- 4. Global alignment ---
        cb("aligning", 55, "Aligning all views into one 3D scene...")
        logger.info(f"Starting global alignment with {n} images")

        if n == 2:
            mode = GlobalAlignerMode.PairViewer
        else:
            mode = GlobalAlignerMode.PointCloudOptimizer

        scene = global_aligner(output, device=self.device, mode=mode)

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            cb("aligning", 60, "Optimizing 3D consistency (200 iterations)...")
            logger.info("Running global alignment optimization...")
            loss = scene.compute_global_alignment(
                init="mst",
                niter=200,
                schedule="cosine",
                lr=0.01
            )
            logger.info(f"Global alignment done, final loss: {loss}")

        cb("aligning", 80, "Alignment complete")

        # --- 5. Extract dense point cloud ---
        cb("extracting", 82, "Extracting dense 3D points and colors...")
        logger.info("Extracting point cloud from aligned scene")

        # Get results
        with torch.no_grad():
            focals = scene.get_focals().detach().cpu().numpy()
            poses = scene.get_im_poses().detach().cpu().numpy()
            pts3d_list = scene.get_pts3d()
            masks = scene.get_masks()
            scene_imgs = scene.imgs

        all_pts = []
        all_cols = []

        for i in range(n):
            # 3D points for each pixel in image i
            p = pts3d_list[i].detach().cpu().numpy()
            logger.info(f"  Image {i}: pts3d shape = {p.shape}")
            p = p.reshape(-1, 3)

            # Confidence mask
            m = masks[i].detach().cpu().numpy().reshape(-1).astype(bool)
            n_valid = m.sum()

            # Colors from image
            img_data = scene_imgs[i]
            if not isinstance(img_data, np.ndarray):
                img_data = np.array(img_data)
            # DUSt3R images are normalized to [0,1] with shape (H, W, 3)
            if img_data.max() <= 1.0:
                c = (img_data * 255).clip(0, 255).astype(np.uint8)
            else:
                c = img_data.clip(0, 255).astype(np.uint8)
            c = c.reshape(-1, 3)

            valid_pts = p[m]
            valid_cols = c[m]

            all_pts.append(valid_pts)
            all_cols.append(valid_cols)

            logger.info(f"  Image {i}: {n_valid} valid points of {len(m)}")
            cb("extracting", 82 + int(6 * (i + 1) / n),
               f"Image {i+1}/{n}: {n_valid} dense points")

        points = np.concatenate(all_pts)
        colors = np.concatenate(all_cols)
        logger.info(f"Total raw dense points: {len(points)}")

        # --- 6. Post-process ---
        cb("extracting", 90, f"Filtering {len(points)} points...")

        # Remove NaN/Inf
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid]
        logger.info(f"After NaN removal: {len(points)}")

        if len(points) > 100:
            # Remove statistical outliers (3σ from median)
            center = np.median(points, axis=0)
            dists = np.linalg.norm(points - center, axis=1)
            threshold = np.percentile(dists, 97)  # Keep 97% of points
            keep = dists < threshold
            points, colors = points[keep], colors[keep]
            logger.info(f"After outlier removal (97th percentile): {len(points)}")

        # Subsample if way too many points for the frontend
        max_points = 500_000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            idx.sort()
            points, colors = points[idx], colors[idx]
            logger.info(f"Subsampled to {len(points)} points")

        # --- 7. Export ---
        cb("exporting", 92, f"Exporting {len(points)} dense 3D points...")

        ply_path = output_dir / "reconstruction.ply"
        self._export_ply(points, colors, ply_path)

        cameras = []
        for i in range(n):
            cameras.append({
                "image": Path(image_paths[i]).name,
                "focal": float(focals[i]),
                "pose": poses[i].tolist(),
            })

        cameras_path = output_dir / "cameras.json"
        with open(cameras_path, "w") as f:
            json.dump(cameras, f, indent=2)

        # Free VRAM
        if self.device == "cuda":
            del scene, output, pts3d_list, masks
            torch.cuda.empty_cache()
            gc.collect()

        cb("done", 100, f"Done! {len(points)} dense 3D points, {len(cameras)} cameras")
        logger.info(f"=== DUSt3R reconstruction complete: {len(points)} points, "
                    f"{len(cameras)} cameras ===")

        return {
            "n_points": len(points),
            "n_cameras": len(cameras),
            "ply_path": str(ply_path),
            "cameras_path": str(cameras_path),
            "cameras": cameras,
        }

    # =========================================================================
    # OpenCV SfM Fallback (sparse)
    # =========================================================================

    def _sfm_reconstruct(self, image_dir: Path, output_dir: Path, progress_callback) -> dict:
        """Sparse SfM fallback using OpenCV SIFT + PnP."""
        cb = progress_callback or (lambda *a: None)

        cb("loading_images", 5, "Loading images (OpenCV SfM fallback)...")
        image_paths = sorted([
            str(p) for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        images_rgb, images_gray, img_sizes = [], [], []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 1600:
                s = 1600 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img_sizes.append((img.shape[1], img.shape[0]))

        n_imgs = len(images_rgb)
        if n_imgs < 2:
            raise ValueError("Need at least 2 images")
        cb("loading_images", 10, f"Loaded {n_imgs} images")

        # SIFT features
        cb("matching", 12, "Detecting features (SIFT)...")
        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.04)
        all_kp, all_desc = [], []
        for gray in images_gray:
            kp, desc = sift.detectAndCompute(gray, None)
            all_kp.append(list(kp))
            all_desc.append(desc)

        # Match pairs
        cb("matching", 17, "Matching features...")
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=150))
        pair_raw_matches = {}

        pairs = [(i, j) for i in range(n_imgs) for j in range(i+1, min(i+6, n_imgs))]
        for i, j in pairs:
            if all_desc[i] is None or all_desc[j] is None: continue
            if len(all_desc[i]) < 10 or len(all_desc[j]) < 10: continue
            raw = flann.knnMatch(all_desc[i], all_desc[j], k=2)
            good = [m for (m, n_) in raw if m.distance < 0.75 * n_.distance]
            if len(good) >= 20:
                pair_raw_matches[(i, j)] = good

        if not pair_raw_matches:
            raise ValueError("No matches found between images")
        cb("matching", 35, f"Matched {len(pair_raw_matches)} pairs")

        # Camera intrinsics
        w0, h0 = img_sizes[0]
        focal = max(w0, h0) * 1.2
        K = np.array([[focal, 0, w0/2], [0, focal, h0/2], [0, 0, 1]], dtype=np.float64)

        # Initial pair
        best_pair = max(pair_raw_matches, key=lambda k: len(pair_raw_matches[k]))
        i0, j0 = best_pair
        matches_init = pair_raw_matches[best_pair]
        pts_i = np.float32([all_kp[i0][m.queryIdx].pt for m in matches_init])
        pts_j = np.float32([all_kp[j0][m.trainIdx].pt for m in matches_init])

        E, mask_E = cv2.findEssentialMat(pts_i, pts_j, K, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
        if E is None:
            raise ValueError("Essential matrix estimation failed")
        mask_E = mask_E.ravel().astype(bool)
        _, R1, t1, _ = cv2.recoverPose(E, pts_i[mask_E], pts_j[mask_E], K)
        t1 = t1.ravel()

        cam_R = {i0: np.eye(3), j0: R1.copy()}
        cam_t = {i0: np.zeros(3), j0: t1.copy()}
        registered: Set[int] = {i0, j0}

        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])

        inlier_idx = np.where(mask_E)[0]
        inlier_matches = [matches_init[idx] for idx in inlier_idx]
        inlier_pts_i = pts_i[mask_E]
        inlier_pts_j = pts_j[mask_E]

        pts4d = cv2.triangulatePoints(P0, P1, inlier_pts_i.T, inlier_pts_j.T)
        pts3d_init = (pts4d[:3] / (pts4d[3:] + 1e-10)).T

        points_3d, point_colors = [], []
        kp_to_3d = {}

        for k in range(len(pts3d_init)):
            pt = pts3d_init[k]
            if pt[2] <= 0 or (R1 @ pt + t1)[2] <= 0 or np.linalg.norm(pt) > 100:
                continue
            proj0 = P0 @ np.append(pt, 1); proj1 = P1 @ np.append(pt, 1)
            if np.linalg.norm(proj0[:2]/proj0[2] - inlier_pts_i[k]) > 4: continue
            if np.linalg.norm(proj1[:2]/proj1[2] - inlier_pts_j[k]) > 4: continue
            idx_3d = len(points_3d)
            points_3d.append(pt)
            m = inlier_matches[k]
            px = int(np.clip(inlier_pts_i[k][0], 0, img_sizes[i0][0]-1))
            py = int(np.clip(inlier_pts_i[k][1], 0, img_sizes[i0][1]-1))
            point_colors.append(images_rgb[i0][py, px])
            kp_to_3d[(i0, m.queryIdx)] = idx_3d
            kp_to_3d[(j0, m.trainIdx)] = idx_3d

        # Incremental PnP
        failed: Set[int] = set()
        for _ in range(n_imgs * 3):
            if len(registered) + len(failed) >= n_imgs: break
            best_img, best_n, best_2d, best_3d_pnp = None, 0, None, None
            for cand in range(n_imgs):
                if cand in registered or cand in failed: continue
                p2d, p3d = [], []
                for reg in registered:
                    key = (min(cand, reg), max(cand, reg))
                    if key not in pair_raw_matches: continue
                    for m in pair_raw_matches[key]:
                        ck, rk = (m.queryIdx, m.trainIdx) if key[0] == cand else (m.trainIdx, m.queryIdx)
                        ri = key[1] if key[0] == cand else key[0]
                        if (ri, rk) in kp_to_3d:
                            p2d.append(all_kp[cand][ck].pt)
                            p3d.append(points_3d[kp_to_3d[(ri, rk)]])
                if len(p2d) > best_n:
                    best_n, best_img = len(p2d), cand
                    best_2d = np.array(p2d, dtype=np.float64)
                    best_3d_pnp = np.array(p3d, dtype=np.float64)
            if best_img is None or best_n < 10: break
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                best_3d_pnp, best_2d, K, None,
                iterationsCount=500, reprojectionError=4.0, confidence=0.99)
            if not ok or inliers is None or len(inliers) < 8:
                failed.add(best_img); continue
            R_new, _ = cv2.Rodrigues(rvec)
            t_new = tvec.ravel()
            cam_R[best_img], cam_t[best_img] = R_new, t_new
            registered.add(best_img)
            # Triangulate new points
            P_new = K @ np.hstack([R_new, t_new.reshape(3, 1)])
            for reg in registered:
                if reg == best_img or reg not in cam_R: continue
                key = (min(best_img, reg), max(best_img, reg))
                if key not in pair_raw_matches: continue
                R_r, t_r = cam_R[reg], cam_t[reg]
                P_r = K @ np.hstack([R_r, t_r.reshape(3, 1)])
                tri_new, tri_reg, tri_idx = [], [], []
                for m in pair_raw_matches[key]:
                    nk, rk = (m.queryIdx, m.trainIdx) if key[0] == best_img else (m.trainIdx, m.queryIdx)
                    if (best_img, nk) not in kp_to_3d and (reg, rk) not in kp_to_3d:
                        tri_new.append(all_kp[best_img][nk].pt)
                        tri_reg.append(all_kp[reg][rk].pt)
                        tri_idx.append((nk, rk))
                if len(tri_new) < 5: continue
                tri_new, tri_reg = np.float32(tri_new), np.float32(tri_reg)
                p4 = cv2.triangulatePoints(P_new, P_r, tri_new.T, tri_reg.T)
                p3 = (p4[:3] / (p4[3:] + 1e-10)).T
                for k2 in range(len(p3)):
                    pt = p3[k2]
                    if (R_new @ pt + t_new)[2] <= 0 or (R_r @ pt + t_r)[2] <= 0: continue
                    if np.linalg.norm(pt) > 100: continue
                    pn = P_new @ np.append(pt, 1); pr = P_r @ np.append(pt, 1)
                    if np.linalg.norm(pn[:2]/pn[2] - tri_new[k2]) > 3: continue
                    if np.linalg.norm(pr[:2]/pr[2] - tri_reg[k2]) > 3: continue
                    idx_3d = len(points_3d)
                    points_3d.append(pt)
                    px_ = int(np.clip(tri_new[k2][0], 0, images_rgb[best_img].shape[1]-1))
                    py_ = int(np.clip(tri_new[k2][1], 0, images_rgb[best_img].shape[0]-1))
                    point_colors.append(images_rgb[best_img][py_, px_])
                    kp_to_3d[(best_img, tri_idx[k2][0])] = idx_3d
                    kp_to_3d[(reg, tri_idx[k2][1])] = idx_3d

        pts, cols = np.array(points_3d), np.array(point_colors)
        if len(pts) < 3:
            raise ValueError("Too few valid 3D points")
        if len(pts) > 50:
            c = np.median(pts, axis=0); d = np.linalg.norm(pts - c, axis=1)
            keep = d < np.mean(d) + 2.5 * np.std(d)
            pts, cols = pts[keep], cols[keep]

        ply_path = output_dir / "reconstruction.ply"
        self._export_ply(pts, cols, ply_path)
        cameras = [{"image": Path(image_paths[idx]).name, "focal": float(focal),
                     "pose": np.hstack([np.vstack([cam_R[idx], [0,0,0]]),
                                       np.append(cam_t[idx], 1).reshape(4,1)]).tolist()}
                    for idx in sorted(cam_R.keys()) if idx in registered]
        cameras_path = output_dir / "cameras.json"
        with open(cameras_path, "w") as f:
            json.dump(cameras, f, indent=2)
        cb("done", 100, f"Done (SfM fallback)! {len(pts)} sparse points")
        return {"n_points": len(pts), "n_cameras": len(cameras),
                "ply_path": str(ply_path), "cameras_path": str(cameras_path), "cameras": cameras}

    # =========================================================================
    # Export
    # =========================================================================

    @staticmethod
    def _export_ply(points, colors, path):
        """Export point cloud as binary PLY."""
        n = len(points)
        if n == 0:
            return
        pts = points.astype(np.float32)
        cols = colors.astype(np.uint8) if colors.dtype != np.uint8 else colors

        header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                  f"property float x\nproperty float y\nproperty float z\n"
                  f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"end_header\n")

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            data = np.zeros(n, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                       ('r','u1'),('g','u1'),('b','u1')])
            data['x'], data['y'], data['z'] = pts[:,0], pts[:,1], pts[:,2]
            data['r'], data['g'], data['b'] = cols[:,0], cols[:,1], cols[:,2]
            f.write(data.tobytes())

        logger.info(f"Exported {n} points to {path}")
