#!/usr/bin/env python3
"""
ctypes bridge to SAM3DBody-cpp (libfast_sam_3dbody.so) + the MHR -> SMPL-22
joint mapping the ROS node publishes.

Deliberately free of ROS imports: the joint mapping and the engine can be
exercised without a sourced ROS 2 workspace (see the self-test at the bottom,
`python3 fsb_engine.py <image-or-video>`).
"""

import ctypes
import os
import sys

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs  (must match src/core/fast_sam_3dbody_capi.h exactly)
# ──────────────────────────────────────────────────────────────────────────────

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",         ctypes.c_char_p),
        ("gguf_path",        ctypes.c_char_p),
        ("yolo_path",        ctypes.c_char_p),
        ("cuda_device",      ctypes.c_int),
        ("skip_body_model",  ctypes.c_int),
        ("person_thresh",    ctypes.c_float),
        ("person_nms_iou",   ctypes.c_float),
        ("max_persons",      ctypes.c_int),
        ("focal_x",          ctypes.c_float),
        ("focal_y",          ctypes.c_float),
        ("principal_x",      ctypes.c_float),
        ("principal_y",      ctypes.c_float),
        ("zero_face_params", ctypes.c_int),
        ("detector",         ctypes.c_int),
        ("refined_pose",     ctypes.c_int),
    ]


class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",             ctypes.c_float * 4),
        ("focal_length",     ctypes.c_float),
        ("pred_cam_t",       ctypes.c_float * 3),
        ("global_rot",       ctypes.c_float * 3),
        ("body_pose",        ctypes.c_float * 133),
        ("shape",            ctypes.c_float * 45),
        ("scale",            ctypes.c_float * 28),
        ("hand_pose",        ctypes.c_float * 108),
        ("face_params",      ctypes.c_float * 72),
        ("yolo_kps",         ctypes.c_float * 51),
        ("has_yolo_kps",     ctypes.c_int),
        ("kps_3d",           ctypes.c_float * 210),   # [70 × 3]
        ("kps_2d",           ctypes.c_float * 140),   # [70 × 2]
        ("has_kps",          ctypes.c_int),
        ("pred_pose_raw",    ctypes.c_float * 266),
        ("pred_cam_raw",     ctypes.c_float * 3),
        ("mhr_model_params", ctypes.c_float * 204),
        ("skel_3d",          ctypes.c_float * 381),   # [127 × 3]
        ("has_skel",         ctypes.c_int),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# MHR -> SMPL-22 joint mapping
#
# D-PoSE (and therefore magician_body_pose_estimation) published
# `joints3d[:, 0:22]`, i.e. the 22 body joints of the SMPL kinematic tree, in
# the canonical SMPL order.  Consumers index that order directly (the node's own
# TF code reads joints[0]=pelvis, [1]=left_hip, [2]=right_hip, [12]=neck), so
# the same order is reproduced here from the MHR outputs:
#
#   * most joints come from the 70 MHR keypoints (FsbResult.kps_3d), using the
#     same renaming as python/fast_sam_3dbody_dump_dpose_compat_csv.py;
#   * pelvis, the spine chain and the head have no MHR-70 keypoint and come from
#     the full 127-joint MHR skeleton (FsbResult.skel_3d).
#
# Indices below are from python/fast_sam_3dbody_dump_csv.py (MHR70) and
# src/core/mhr_joint_table.h (MHR127, verified against its PARENTS
# table: root(1) -> c_spine0(34) -> c_spine1(35) -> c_spine2(36) -> c_spine3(37)
# -> c_neck(110) -> c_head(113)).
# ──────────────────────────────────────────────────────────────────────────────

SMPL22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

# Parent of each SMPL-22 joint (-1 for the root) — used to draw the overlay.
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                  16, 17, 18, 19]

# (SMPL-22 index, MHR-70 keypoint index)
_FROM_KPS = [
    (1,   9),   # left_hip        <- left_hip
    (2,  10),   # right_hip       <- right_hip
    (4,  11),   # left_knee       <- left_knee
    (5,  12),   # right_knee      <- right_knee
    (7,  13),   # left_ankle      <- left_ankle
    (8,  14),   # right_ankle     <- right_ankle
    (10, 15),   # left_foot       <- left_big_toe_tip
    (11, 18),   # right_foot      <- right_big_toe_tip
    (12, 69),   # neck            <- neck
    (13, 67),   # left_collar     <- left_acromion
    (14, 68),   # right_collar    <- right_acromion
    (16,  5),   # left_shoulder   <- left_shoulder
    (17,  6),   # right_shoulder  <- right_shoulder
    (18,  7),   # left_elbow      <- left_elbow
    (19,  8),   # right_elbow     <- right_elbow
    (20, 62),   # left_wrist      <- left_wrist
    (21, 41),   # right_wrist     <- right_wrist
]

# (SMPL-22 index, MHR-127 skeleton joint index)
_FROM_SKEL = [
    (0,    1),  # pelvis  <- root
    (3,   35),  # spine1  <- c_spine1
    (6,   36),  # spine2  <- c_spine2
    (9,   37),  # spine3  <- c_spine3
    (15, 113),  # head    <- c_head
]

_KPS_DST  = np.array([d for d, _ in _FROM_KPS])
_KPS_SRC  = np.array([s for _, s in _FROM_KPS])
_SKEL_DST = np.array([d for d, _ in _FROM_SKEL])
_SKEL_SRC = np.array([s for _, s in _FROM_SKEL])


def smpl22_joints(result):
    """
    SMPL-22 joints of one person, in the camera frame, as a (22, 3) array.

    OpenCV camera axes (x right, y down, z forward), metres — the frame
    magician_body_pose_estimation published and the one its ArUco maths assumes.
    `kps_3d` / `skel_3d` are body-centred, so the camera translation is added
    here, exactly like D-PoSE's `joints3d + pred_cam_t`.
    """
    kps  = np.array(result.kps_3d,  dtype=np.float64).reshape(70, 3)
    skel = np.array(result.skel_3d, dtype=np.float64).reshape(127, 3)

    joints = np.empty((22, 3), dtype=np.float64)
    joints[_KPS_DST]  = kps[_KPS_SRC]
    joints[_SKEL_DST] = skel[_SKEL_SRC]
    return joints + np.array(result.pred_cam_t, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

LIB_NAME = "libfast_sam_3dbody.so"


def repo_root_from(path):
    """
    SAM3DBody-cpp checkout this package lives in, or None.

    The package sits at <repo>/src/ros2/poseEstimation3D, and realpath() makes
    that work through the symlink that puts it in a ROS 2 workspace.  The copy
    colcon installs sits under <ws>/install instead, from where the checkout
    cannot be found that way — so the build writes engine_root.py next to it,
    holding the path the package was built from.
    """
    pkg = os.path.dirname(os.path.realpath(path))
    root = os.path.realpath(os.path.join(pkg, "..", "..", ".."))
    if os.path.isfile(os.path.join(root, "CMakeLists.txt")):
        return root
    try:
        from engine_root import ENGINE_ROOT
    except ImportError:
        return None
    return ENGINE_ROOT if os.path.isdir(ENGINE_ROOT) else None


class Engine:
    """One loaded SAM-3D-Body pipeline."""

    def __init__(self, lib_dir, onnx_dir, gguf_path, yolo_path, cuda_device=0,
                 person_thresh=0.5, person_nms_iou=0.45, max_persons=0,
                 fx=0.0, fy=0.0, cx=0.0, cy=0.0, refined_pose=False):
        self._lib = _load_library(lib_dir)
        self._handle = self._lib.fsb_create()
        if not self._handle:
            raise RuntimeError("fsb_create() returned NULL")

        cfg = FsbConfig(
            onnx_dir         = onnx_dir.encode(),
            gguf_path        = gguf_path.encode(),
            yolo_path        = yolo_path.encode(),
            cuda_device      = cuda_device,
            skip_body_model  = 0,          # we need kps_3d / skel_3d
            person_thresh    = person_thresh,
            person_nms_iou   = person_nms_iou,
            max_persons      = max_persons,
            focal_x          = fx,
            focal_y          = fy,
            principal_x      = cx,
            principal_y      = cy,
            zero_face_params = 0,
            detector         = 0,          # YOLO11-pose
            refined_pose     = 1 if refined_pose else 0,
        )
        if not self._lib.fsb_load(self._handle, ctypes.byref(cfg)):
            self._lib.fsb_destroy(self._handle)
            self._handle = None
            raise RuntimeError(
                "fsb_load() failed — see the [FSB] lines above. With "
                "--refined-pose this usually means the 'refined' model profile "
                "is missing (bash tools/fetch_model.sh cuda refined)."
            )

        capacity = max_persons if max_persons > 0 else 32
        self._results = (FsbResult * capacity)()
        self._capacity = capacity

    def process(self, frame_bgr):
        """Run the pipeline on one BGR frame; return the per-person results."""
        frame_bgr = np.ascontiguousarray(frame_bgr)
        height, width = frame_bgr.shape[:2]
        n = self._lib.fsb_process_bgr(
            self._handle,
            frame_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width, height,
            self._results, self._capacity,
        )
        return [self._results[i] for i in range(n)]

    def close(self):
        if self._handle:
            self._lib.fsb_destroy(self._handle)
            self._handle = None


def _load_library(lib_dir):
    lib_path = os.path.join(lib_dir, LIB_NAME)
    if not os.path.exists(lib_path):
        raise RuntimeError(f"{lib_path} not found — build SAM3DBody-cpp first "
                           f"(bash setup.sh in this package).")

    # ONNX Runtime ships next to the library; ld.so has to find it too.
    ort_lib = os.path.join(lib_dir, "onnxruntime_dl", "lib")
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in (lib_dir, ort_lib, previous) if p)

    lib = ctypes.CDLL(lib_path)
    lib.fsb_result_size.restype  = ctypes.c_int
    lib.fsb_result_size.argtypes = []
    if lib.fsb_result_size() != ctypes.sizeof(FsbResult):
        raise RuntimeError(
            f"FsbResult layout mismatch: C library has {lib.fsb_result_size()} bytes, "
            f"this script {ctypes.sizeof(FsbResult)} — update the ctypes mirror "
            f"to match src/core/fast_sam_3dbody_capi.h")
    lib.fsb_create.restype = ctypes.c_void_p
    lib.fsb_create.argtypes = []
    lib.fsb_destroy.restype = None
    lib.fsb_destroy.argtypes = [ctypes.c_void_p]
    lib.fsb_load.restype = ctypes.c_int
    lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]
    lib.fsb_process_bgr.restype = ctypes.c_int
    lib.fsb_process_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(FsbResult),
        ctypes.c_int,
    ]
    return lib


# ──────────────────────────────────────────────────────────────────────────────
# Self-test: run the engine on one frame and print the SMPL-22 joints.
# No ROS needed — this is what verifies the joint mapping.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2

    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <image|video> [--refined-pose]")

    root = repo_root_from(__file__)
    if root is None:
        sys.exit("Could not locate the SAM3DBody-cpp checkout from this file.")

    source = sys.argv[1]
    frame = cv2.imread(source)
    if frame is None:
        capture = cv2.VideoCapture(source)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            sys.exit(f"Could not read a frame from {source}")

    onnx = os.path.join(root, "onnx")
    engine = Engine(
        lib_dir   = os.path.join(root, "build"),
        onnx_dir  = onnx,
        gguf_path = os.path.join(onnx, "pipeline.gguf"),
        yolo_path = os.path.join(onnx, "yolo.onnx"),
        refined_pose = "--refined-pose" in sys.argv,
    )
    for person, result in enumerate(engine.process(frame)):
        if not (result.has_kps and result.has_skel):
            print(f"person {person}: no body model output")
            continue
        joints = smpl22_joints(result)
        print(f"person {person}  bbox={list(result.bbox)}")
        for index, name in enumerate(SMPL22_NAMES):
            print(f"  {index:2d} {name:15s} "
                  f"{joints[index][0]: .3f} {joints[index][1]: .3f} "
                  f"{joints[index][2]: .3f}")
    engine.close()
