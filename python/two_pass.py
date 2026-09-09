"""
two_pass.py  –  Second-pass Python model runner for fast_sam_3dbody_frontend.py

Architecture
────────────
The C++ pipeline runs one decoder pass: backbone → decoder → MHR-FFN → kps_3d.
The Python SAM-3D-Body model runs *two* passes: on the second pass the decoder is
conditioned on both the first-pass pose estimate (prev_estimate) and YOLO-derived
keypoint prompts, producing geometrically consistent 3-D joint positions that match
the observed 2-D skeleton.  This is why the Python demo_webcam.py has better hand
placement, head orientation, and no ghost-limb artefacts.

This module wraps the full Python estimator.  When --two-passes is requested the
frontend:
  1. Still runs the C++ first pass (fast, provides YOLO bboxes).
  2. Calls SecondPassRunner.run(frame_bgr, cpp_results) once per frame.
  3. Uses the returned kps_2d[70,2] for visualisation; falls back to C++ kps_2d on failure.

The Python model re-runs the backbone (redundant with C++, unavoidable with the
current architecture because forward_decoder needs PyTorch backbone features, not
ONNX features).  The latency cost is roughly equal to one full C++ pass.

Usage
─────
    from two_pass import SecondPassRunner, SecondPassConfig

    cfg = SecondPassConfig(
        checkpoint   = "checkpoints/sam-3d-body-dinov3/model.ckpt",
        mhr_model    = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        device       = "cuda",   # or "cpu"
    )
    runner = SecondPassRunner(cfg)        # load once at startup

    # per-frame:
    kps_list = runner.run(frame_bgr, cpp_results)   # list[np.ndarray|None] len==n_persons
    for i, kps in enumerate(kps_list):
        if kps is not None:
            draw_mhr70(vis, cpp_results[i], kps_override=kps)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports – only loaded when SecondPassRunner is constructed, so the rest
# of fast_sam_3dbody_frontend.py remains importable even without PyTorch.
# ──────────────────────────────────────────────────────────────────────────────

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("[two_pass] PyTorch is required for --two-passes.  "
                 "Install with: pip install torch")

def _require_sam3d():
    try:
        from sam_3d_body import load_sam_3d_body
        from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
        return load_sam_3d_body, SAM3DBodyEstimator
    except ImportError:
        sys.exit("[two_pass] sam_3d_body package not found.  "
                 "Make sure the sam-3d-body repo is on PYTHONPATH.")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SecondPassConfig:
    # Path to model.ckpt (the main checkpoint, must be in the same dir as
    # model_config.yaml — that's how load_sam_3d_body locates the config).
    checkpoint: str = ""

    # Path to the MHR body model (mhr_model.pt).
    mhr_model: str = ""

    # PyTorch device.  "cuda" requires CUDA; falls back to "cpu" automatically
    # if CUDA is unavailable.
    device: str = "cuda"

    # Inference type passed to process_one_image.  "full" runs both body and
    # hand decoders.  "body" is faster (body only).
    inference_type: str = "full"


# ──────────────────────────────────────────────────────────────────────────────
# SecondPassRunner
# ──────────────────────────────────────────────────────────────────────────────

class SecondPassRunner:
    """Loads the Python SAM-3D-Body model and runs the second decoder pass."""

    def __init__(self, cfg: SecondPassConfig):
        torch = _require_torch()
        load_sam_3d_body, SAM3DBodyEstimator = _require_sam3d()

        if not cfg.checkpoint:
            raise ValueError("[two_pass] SecondPassConfig.checkpoint must be set")

        # HIGH RISK: sam_3d_body_estimator.py line 328 hardcodes
        #   batch = recursive_to(batch, "cuda")
        # so the inference batch ALWAYS runs on CUDA regardless of what device is
        # passed to load_sam_3d_body.  If the model is on CPU but the batch is on
        # CUDA, data_preprocess crashes with a device mismatch on image_mean.
        # Resolution:
        #   • When CUDA is available — ALWAYS use CUDA (matches estimator's assumption).
        #   • When CUDA is absent   — keep CPU (recursive_to to "cuda" becomes a no-op
        #     on a CPU-only build of PyTorch, so CPU inference works there).
        if cfg.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("[two_pass] CUDA not available, falling back to CPU "
                          "(second pass will be slower)")
            cfg.device = "cpu"
        elif cfg.device == "cpu" and torch.cuda.is_available():
            warnings.warn(
                "[two_pass] sam_3d_body_estimator always moves the inference batch "
                "to CUDA (hardcoded).  Overriding device='cpu' to 'cuda' to avoid "
                "the device mismatch in data_preprocess.  To run purely on CPU, "
                "use a machine without a CUDA-capable GPU."
            )
            cfg.device = "cuda"

        print(f"[two_pass] Loading Python model from {cfg.checkpoint} …")
        t0 = time.perf_counter()

        # load_sam_3d_body reads model_config.yaml from the same directory as
        # the checkpoint.  It returns (model, model_cfg).
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            mhr_path=cfg.mhr_model,
        )

        # SAM3DBodyEstimator wraps the model for top-down inference.
        # We pass no human_detector — bboxes will be provided by C++ YOLO.
        self._estimator = SAM3DBodyEstimator(model, model_cfg)
        self._inference_type = cfg.inference_type
        self._device = cfg.device

        elapsed = time.perf_counter() - t0
        print(f"[two_pass] Model loaded in {elapsed:.1f}s on {cfg.device}")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, frame_bgr: np.ndarray, cpp_results) -> List[Optional[np.ndarray]]:
        """Run the second decoder pass for all detected persons.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Full frame in BGR uint8 (OpenCV default).
        cpp_results : list[FsbResult]
            Per-person results from the C++ first pass.

        Returns
        -------
        list of np.ndarray or None
            One entry per person.  Each entry is either:
              • np.ndarray shape [70,2] float32 — 2-D keypoints in image pixels
              • None — second pass failed; caller should fall back to C++ kps_2d.
        """
        if not cpp_results:
            return []

        # Collect bboxes from C++ results: shape [N, 4] float32 [x1,y1,x2,y2]
        bboxes = np.array(
            [[r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]] for r in cpp_results],
            dtype=np.float32,
        )

        # The Python model expects RGB (not BGR).
        # HIGH RISK: if we pass BGR the backbone features will be wrong — the
        # model was trained on RGB.  cv2.cvtColor is a simple channel swap.
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            outputs = self._estimator.process_one_image(
                img_rgb,
                bboxes=bboxes,
                inference_type=self._inference_type,
            )
        except Exception as exc:
            warnings.warn(f"[two_pass] process_one_image failed: {exc}")
            return [None] * len(cpp_results)

        # process_one_image returns one dict per person.  The number of returned
        # persons may differ from len(cpp_results) if the model merges or drops
        # persons — guard against index mismatch.
        results: List[Optional[np.ndarray]] = [None] * len(cpp_results)

        for i, out in enumerate(outputs):
            if i >= len(cpp_results):
                break
            kps = _extract_kps2d(out)
            results[i] = kps

        return results

    # ── Smoke-test ────────────────────────────────────────────────────────────

    def smoke_test(self, frame_bgr: np.ndarray) -> bool:
        """Run inference on a single dummy person covering the full frame.

        Returns True if inference completed without error and produced a
        non-empty, finite kps_2d array.  Used by TestPhase1 in test_two_pass.py.
        """
        H, W = frame_bgr.shape[:2]
        # Fake a single bbox covering the whole image
        fake_bbox = np.array([[0, 0, W, H]], dtype=np.float32)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            outputs = self._estimator.process_one_image(
                img_rgb,
                bboxes=fake_bbox,
                inference_type="body",   # faster for smoke-test
            )
        except Exception as exc:
            warnings.warn(f"[two_pass] smoke_test failed: {exc}")
            return False

        if not outputs:
            return False
        kps = _extract_kps2d(outputs[0])
        if kps is None:
            return False
        return bool(np.isfinite(kps).all())


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_kps2d(out: dict) -> Optional[np.ndarray]:
    """Extract 2-D keypoints from a process_one_image output dict.

    Returns float32 array of shape [70, 2] (image pixel coords), or None.

    The Python model's pred_keypoints_2d has shape [K, 2] where K is the number
    of MHR keypoints (70 for MHR70).  Values are pixel coordinates in the
    original (full) image space, matching the C++ kps_2d layout.
    """
    kps = out.get("pred_keypoints_2d", None)
    if kps is None:
        return None

    # Handle both torch.Tensor and numpy array
    try:
        import torch as _torch
        if isinstance(kps, _torch.Tensor):
            kps = kps.cpu().numpy()
    except ImportError:
        pass

    kps = np.asarray(kps, dtype=np.float32)

    # Expected shape: [70, 2].  Silently reject unexpected shapes.
    if kps.ndim != 2 or kps.shape[1] < 2:
        warnings.warn(f"[two_pass] unexpected pred_keypoints_2d shape {kps.shape}")
        return None

    # Keep only x,y columns (column 2, if present, is confidence)
    kps = kps[:, :2]

    if not np.isfinite(kps).all():
        warnings.warn("[two_pass] pred_keypoints_2d contains NaN/Inf — dropping")
        return None

    return kps


def _build_prev_estimate_from_result(result, has_init_camera: bool) -> "np.ndarray":
    """Build prev_estimate[1,1,519+] from a C++ FsbResult.

    This mirrors run_keypoint_prompt() in sam3d_body.py lines 1654-1668:

        prev_estimate = cat([pred_pose_raw, shape, scale, hand, face], dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = cat([prev_estimate, pred_cam], dim=-1)

    Layout of the output array (axis -1):
        [0:266]   pred_pose_raw (global_rot_6d[6] + body_cont[260])
        [266:311] shape[45]
        [311:339] scale[28]
        [339:447] hand_pose[108]
        [447:519] face_params[72]
        [519:522] pred_cam_raw[3]  (only if has_init_camera=True)

    Returns shape [1, 1, 519] or [1, 1, 522].

    HIGH RISK: The layout must match the Python model's prev_to_token_mhr linear
    layer exactly.  Any offset shifts the entire embedding → wrong decoder output.
    If this function returns garbage, the second pass will produce worse results
    than the first pass.  Test with TestPhase2FsbResult.test_prev_estimate_layout.
    """
    parts = [
        np.array(result.pred_pose_raw, dtype=np.float32),  # [266]
        np.array(result.shape,         dtype=np.float32),  # [45]
        np.array(result.scale,         dtype=np.float32),  # [28]
        np.array(result.hand_pose,     dtype=np.float32),  # [108]
        np.array(result.face_params,   dtype=np.float32),  # [72]
    ]
    if has_init_camera:
        parts.append(np.array(result.pred_cam_raw, dtype=np.float32))  # [3]

    flat = np.concatenate(parts)       # [519] or [522]
    return flat[np.newaxis, np.newaxis, :]   # [1, 1, 519+]
