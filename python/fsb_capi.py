#!/usr/bin/env python3
"""
fsb_capi.py — the ctypes binding of libfast_sam_3dbody.so (src/core/fast_sam_3dbody_capi.h).

The scripts in python/ import FsbConfig / FsbResult / load_library from here, so
there is one ctypes mirror of the C structs to keep in step with the header
(python/test_ctypes_layout.py checks it against the built library).

Low level, as the scripts use it:

    from fsb_capi import FsbConfig, FsbResult, load_library
    lib = load_library("build")                 # checks fsb_result_size()
    h = lib.fsb_create(); lib.fsb_load(h, ctypes.byref(FsbConfig(...)))
    n = lib.fsb_process_bgr(h, bgr_ptr, W, H, (FsbResult * 32)(), 32)

High level, numpy in / numpy out:

    from fsb_capi import FastSAM3DBody
    with FastSAM3DBody(lib_dir="build", onnx_dir="onnx", refined_pose=True) as fsb:
        people = fsb.process(bgr)               # list of dicts, one per person
        people = fsb.process(bgr, boxes=[[x1, y1, x2, y2]], iou_skip=0.5)   # + your own person boxes

See knowledge/OUTPUT.md for what each field holds.
"""
import ctypes
import os
import sys

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs matching fast_sam_3dbody_capi.h
# ──────────────────────────────────────────────────────────────────────────────

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",       ctypes.c_char_p),
        ("gguf_path",      ctypes.c_char_p),
        ("yolo_path",      ctypes.c_char_p),
        ("cuda_device",    ctypes.c_int),
        ("skip_body_model",ctypes.c_int),
        ("person_thresh",  ctypes.c_float),
        ("person_nms_iou", ctypes.c_float),
        ("max_persons",    ctypes.c_int),
        ("focal_x",        ctypes.c_float),
        ("focal_y",        ctypes.c_float),
        ("principal_x",    ctypes.c_float),
        ("principal_y",    ctypes.c_float),
        ("zero_face_params", ctypes.c_int),  # 0/1 — force face expression to neutral
        ("detector",         ctypes.c_int),
        ("refined_pose",     ctypes.c_int),
    ]

class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",         ctypes.c_float * 4),
        ("focal_length", ctypes.c_float),
        ("pred_cam_t",   ctypes.c_float * 3),
        ("global_rot",   ctypes.c_float * 3),
        ("body_pose",    ctypes.c_float * 133),
        ("shape",        ctypes.c_float * 45),
        ("scale",        ctypes.c_float * 28),
        ("hand_pose",    ctypes.c_float * 108),
        ("face_params",  ctypes.c_float * 72),
        ("yolo_kps",     ctypes.c_float * 51),
        ("has_yolo_kps", ctypes.c_int),
        ("kps_3d",       ctypes.c_float * 210),
        ("kps_2d",       ctypes.c_float * 140),
        ("has_kps",      ctypes.c_int),
        # ── Second-pass raw fields (must stay at end — appended after v1 ABI) ──
        # pred_pose_raw[266]: raw MHR FFN output, layout global_rot_6d[6] + body_cont[260].
        # pred_cam_raw[3]:    raw cam FFN output [s, tx, ty] before nonlinear decode.
        # Both are consumed by two_pass.py to build prev_estimate for forward_decoder.
        # HIGH RISK: any offset here shifts ALL ctypes reads for this struct.
        ("pred_pose_raw", ctypes.c_float * 266),
        ("pred_cam_raw",  ctypes.c_float * 3),
        # mhr_model_params[204]: assembled model_params used by native C LBS.
        # Layout: [0:3]=global_trans*10, [3:6]=global_rot ZYX, [6:136]=body_pose[:130],
        #         [136:204]=scale_out.  Mirrors Python mhr_forward(return_model_params=True).
        ("mhr_model_params", ctypes.c_float * 204),
        ("skel_3d",          ctypes.c_float * 381),   # [127 × 3]
        ("has_skel",         ctypes.c_int),
    ]


# COCO-17 joint k is MHR-70 joint COCO_TO_MHR70[k] (MHR-70 wrists are 62 left / 41 right).
# The MHR-70 hips (9 / 10) are the anatomical hip joints, which sit inside COCO's
# annotated hips: see knowledge/OUTPUT.md before comparing them with COCO data.
COCO_TO_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def load_library(lib_dir: str) -> ctypes.CDLL:
    lib_path = os.path.join(lib_dir, "libfast_sam_3dbody.so")
    if not os.path.exists(lib_path):
        sys.exit(f"Library not found: {lib_path}\nBuild the project first.")

    # Add the lib directory to LD_LIBRARY_PATH so transitive .so deps are found
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    ort_lib = os.path.join(lib_dir, "onnxruntime_dl", "lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [lib_dir, ort_lib, prev]))

    lib = ctypes.CDLL(lib_path)
    lib.fsb_result_size.restype  = ctypes.c_int
    lib.fsb_result_size.argtypes = []
    if lib.fsb_result_size() != ctypes.sizeof(FsbResult):
        raise RuntimeError(
            f"FsbResult layout mismatch: C library has {lib.fsb_result_size()} bytes, "
            f"this script {ctypes.sizeof(FsbResult)} — update the ctypes mirror "
            f"to match src/core/fast_sam_3dbody_capi.h")

    lib.fsb_create.restype  = ctypes.c_void_p
    lib.fsb_create.argtypes = []

    lib.fsb_destroy.restype  = None
    lib.fsb_destroy.argtypes = [ctypes.c_void_p]

    lib.fsb_load.restype  = ctypes.c_int
    lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]

    lib.fsb_process_bgr.restype  = ctypes.c_int
    lib.fsb_process_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(FsbResult),
        ctypes.c_int,
    ]

    # Older builds lack fsb_process_bgr_boxes(); FastSAM3DBody.process(boxes=...) then raises.
    if hasattr(lib, "fsb_process_bgr_boxes"):
        lib.fsb_process_bgr_boxes.restype  = ctypes.c_int
        lib.fsb_process_bgr_boxes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),   # boxes: n_boxes × [x1, y1, x2, y2]
            ctypes.c_int,                     # n_boxes
            ctypes.c_float,                   # iou_skip
            ctypes.POINTER(FsbResult),
            ctypes.c_int,
        ]
    return lib


# ──────────────────────────────────────────────────────────────────────────────
# numpy view of a result
# ──────────────────────────────────────────────────────────────────────────────

# Field shapes for result_to_dict(); fields not listed stay flat.
SHAPES = {
    "yolo_kps": (17, 3),
    "kps_3d":   (70, 3),
    "kps_2d":   (70, 2),
    "skel_3d":  (127, 3),
}


def result_to_dict(r: FsbResult) -> dict:
    """FsbResult -> {field: numpy float32 array (copied) | float | int}; see SHAPES."""
    out = {}
    for name, ctype in FsbResult._fields_:
        v = getattr(r, name)
        if isinstance(v, ctypes.Array):
            a = np.ctypeslib.as_array(v).astype(np.float32)   # astype copies: the buffer is reused
            out[name] = a.reshape(SHAPES[name]) if name in SHAPES else a
        else:
            out[name] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# High-level wrapper
# ──────────────────────────────────────────────────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FastSAM3DBody:
    """One loaded pipeline. Defaults match python/fast_sam_3dbody_frontend.py."""

    def __init__(self, lib_dir: str = os.path.join(_REPO, "build"),
                 onnx_dir: str = os.path.join(_REPO, "onnx"),
                 gguf_path: str = "", yolo_path: str = "",
                 cuda_device: int = 0, person_thresh: float = 0.5, person_nms_iou: float = 0.45,
                 max_persons: int = 0, focal_x: float = 0.0, focal_y: float = 0.0,
                 principal_x: float = 0.0, principal_y: float = 0.0,
                 skip_body_model: bool = False, zero_face_params: bool = True,
                 detector: int = 0, refined_pose: bool = False, max_results: int = 64):
        self.lib = load_library(lib_dir)
        self.handle = self.lib.fsb_create()
        if not self.handle:
            raise RuntimeError("fsb_create() returned NULL")
        self.cfg = FsbConfig(
            onnx_dir         = onnx_dir.encode(),
            gguf_path        = (gguf_path or os.path.join(onnx_dir, "pipeline.gguf")).encode(),
            yolo_path        = (yolo_path or os.path.join(onnx_dir, "yolo.onnx")).encode(),
            cuda_device      = cuda_device,
            skip_body_model  = int(skip_body_model),
            person_thresh    = person_thresh,
            person_nms_iou   = person_nms_iou,
            max_persons      = max_persons,
            focal_x          = focal_x,
            focal_y          = focal_y,
            principal_x      = principal_x,
            principal_y      = principal_y,
            zero_face_params = int(zero_face_params),
            detector         = detector,
            refined_pose     = int(refined_pose),
        )
        if not self.lib.fsb_load(self.handle, ctypes.byref(self.cfg)):
            self.close()
            raise RuntimeError(f"fsb_load() failed (models in {onnx_dir}?)")
        self.max_results = max(max_results, max_persons)
        self.results = (FsbResult * self.max_results)()

    def process_raw(self, bgr: np.ndarray, boxes=None, iou_skip: float = 0.5) -> int:
        """Run on a BGR uint8 H×W×3 image; returns n, the results are self.results[:n] (overwritten by the
        next call). boxes: n × [x1, y1, x2, y2] person boxes of your own, in image pixels, regressed too
        unless a detector box overlaps one with IoU >= iou_skip (iou_skip > 1: always; < 0: only your
        boxes, no detector boxes); their results have has_yolo_kps = 0 and bbox = the box as given."""
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"expected an H×W×3 BGR image, got shape {bgr.shape}")
        H, W = bgr.shape[:2]
        ptr = bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if boxes is None:
            return self.lib.fsb_process_bgr(self.handle, ptr, W, H, self.results, self.max_results)
        if not hasattr(self.lib, "fsb_process_bgr_boxes"):
            raise RuntimeError("this libfast_sam_3dbody.so has no fsb_process_bgr_boxes(): rebuild it")
        b = np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 4)
        return self.lib.fsb_process_bgr_boxes(self.handle, ptr, W, H,
                                              b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(b),
                                              float(iou_skip), self.results, self.max_results)

    def process(self, bgr: np.ndarray, boxes=None, iou_skip: float = 0.5) -> list:
        """As process_raw(), but returns a list of result_to_dict() dicts, one per person."""
        n = self.process_raw(bgr, boxes, iou_skip)
        return [result_to_dict(self.results[i]) for i in range(n)]

    def close(self):
        if self.handle:
            self.lib.fsb_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
