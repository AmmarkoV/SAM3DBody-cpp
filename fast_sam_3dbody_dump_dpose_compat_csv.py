#!/usr/bin/env python3
"""
fast_sam_3dbody_dump_dpose_compat_csv.py
Runs the C++ SAM-3D-Body pipeline and dumps per-frame 3D joints to CSV using the
SAME joint names / layout as D-PoSE's (https://github.com/AmmarkoV/D-PoSE/blob/main/demo_webcam.py) demo_webcam_csv.py
(encode_smplx_skeleton_to_dict), so the output is drop-in compatible with the
feature-recognition tooling that already consumes D-PoSE CSVs.

WHY a separate frontend (vs fast_sam_3dbody_dump_csv.py):
  - dump_csv.py emits the native 70 MHR keypoint names (nose, left_eye, …).
  - D-PoSE emits SMPL-X joint names (pelvis, spine1/2/3, left_hand_thumb1, …).
  This script bridges the two: it renames the MHR joints to the SMPL-X names and
  pulls pelvis + spine1/2/3 (absent from the 70 keypoints) from the full 127-joint
  MHR skeleton now exposed by the C API (FsbResult.skel_3d / has_skel).

Joint set mirrors D-PoSE encode_smplx_skeleton_to_dict:
  22 SMPL body joints + 21 left-hand + 21 right-hand + head landmarks.

CSV format mirrors D-PoSE write_skeleton_rows_to_csv:
  header:  frame_id,skeleton_id,<joint>_x,<joint>_y,<joint>_z,...
  rows:    one row per (frame, skeleton); joint columns sorted for stability.

Usage (kept close to D-PoSE demo_webcam_csv.py):
  python fast_sam_3dbody_dump_dpose_compat_csv.py --input /dev/video0 --save
  python fast_sam_3dbody_dump_dpose_compat_csv.py --input video.mp4 --output_folder out
  python fast_sam_3dbody_dump_dpose_compat_csv.py --input image.jpg --display
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# MHR70 joint index → name  (from sam_3d_body/metadata/mhr70.py original_keypoint_info)
# ──────────────────────────────────────────────────────────────────────────────

MHR70_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint", "right_thumb_third_joint",
    "right_index_tip", "right_index_first_joint", "right_index_second_joint", "right_index_third_joint",
    "right_middle_tip", "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint", "right_ring_third_joint",
    "right_pinky_tip", "right_pinky_first_joint", "right_pinky_second_joint", "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint", "left_thumb_third_joint",
    "left_index_tip", "left_index_first_joint", "left_index_second_joint", "left_index_third_joint",
    "left_middle_tip", "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint", "left_ring_third_joint",
    "left_pinky_tip", "left_pinky_first_joint", "left_pinky_second_joint", "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]
MHR70_IDX = {n: i for i, n in enumerate(MHR70_NAMES)}

# ──────────────────────────────────────────────────────────────────────────────
# MHR 127-joint skeleton index → name  (from src/mhr_joint_table.h)
# Only the joints we actually read are listed here for clarity.
# ──────────────────────────────────────────────────────────────────────────────

MHR127_IDX = {
    "root":     1,   # ≈ pelvis
    "c_spine0": 33,
    "c_spine1": 34,
    "c_spine2": 35,
    "c_spine3": 36,
}

# ──────────────────────────────────────────────────────────────────────────────
# SMPL-X body joint mapping (D-PoSE SMPL_JOINT_NAMES order).
# value = MHR70 source name, or a callable for derived joints.
# pelvis / spine1 / spine2 / spine3 come from the 127-joint skeleton.
# ──────────────────────────────────────────────────────────────────────────────

# Direct MHR70 → SMPL-X body name renames.
SMPLX_BODY_FROM_MHR70 = {
    "left_hip":       "left_hip",
    "right_hip":      "right_hip",
    "left_knee":      "left_knee",
    "right_knee":     "right_knee",
    "left_ankle":     "left_ankle",
    "right_ankle":    "right_ankle",
    "left_foot":      "left_big_toe_tip",
    "right_foot":     "right_big_toe_tip",
    "neck":           "neck",
    "left_collar":    "left_acromion",
    "right_collar":   "right_acromion",
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow":     "left_elbow",
    "right_elbow":    "right_elbow",
    "left_wrist":     "left_wrist",
    "right_wrist":    "right_wrist",
}

# SMPL-X body joints filled from the 127-joint skeleton (true spine chain).
SMPLX_BODY_FROM_SKEL = {
    "pelvis": "root",
    "spine1": "c_spine1",
    "spine2": "c_spine2",
    "spine3": "c_spine3",
}

# D-PoSE hand naming: {side}_hand_{name}, MANO order = wrist, then 4 per finger
# base→tip (thumb1..thumb4, index1..index4, ...).  MHR70 stores each finger
# tip-first (tip, first, second, third) with the wrist last; we reorder to
# match D-PoSE base→tip so downstream indexing is identical.
HAND_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _mhr_finger_basetotip(side: str, finger: str):
    """MHR70 names for a finger, ordered base→tip (joint1..joint4)."""
    return [
        f"{side}_{finger}_third_joint",   # base   -> {finger}1
        f"{side}_{finger}_second_joint",  #        -> {finger}2
        f"{side}_{finger}_first_joint",   #        -> {finger}3
        f"{side}_{finger}_tip",           # tip    -> {finger}4
    ]


def _build_hand_map():
    """{dpose_name: mhr70_name} for both hands, MANO base→tip order."""
    m = {}
    for side in ("left", "right"):
        m[f"{side}_hand_wrist"] = f"{side}_wrist"
        for finger in HAND_FINGERS:
            mhr_names = _mhr_finger_basetotip(side, finger)
            for i, mhr_name in enumerate(mhr_names, start=1):
                m[f"{side}_hand_{finger}{i}"] = mhr_name
    return m


HAND_MAP = _build_hand_map()

# D-PoSE head landmarks come from the model's dedicated head_3d output, which the
# C pipeline does not expose separately.  We approximate the subset that maps
# cleanly onto MHR70 facial keypoints so the column names still line up.
HEAD_MAP = {
    "head_nose":      "nose",
    "head_left_eye":  "left_eye",
    "head_right_eye": "right_eye",
    "head_left_ear":  "left_ear",
    "head_right_ear": "right_ear",
}

# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs  (must match fast_sam_3dbody_capi.h exactly)
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
    ]


class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",            ctypes.c_float * 4),
        ("focal_length",    ctypes.c_float),
        ("pred_cam_t",      ctypes.c_float * 3),
        ("global_rot",      ctypes.c_float * 3),
        ("body_pose",       ctypes.c_float * 133),
        ("shape",           ctypes.c_float * 45),
        ("scale",           ctypes.c_float * 28),
        ("hand_pose",       ctypes.c_float * 108),
        ("face_params",     ctypes.c_float * 72),
        ("yolo_kps",        ctypes.c_float * 51),
        ("has_yolo_kps",    ctypes.c_int),
        ("kps_3d",          ctypes.c_float * 210),
        ("kps_2d",          ctypes.c_float * 140),
        ("has_kps",         ctypes.c_int),
        ("pred_pose_raw",   ctypes.c_float * 266),
        ("pred_cam_raw",    ctypes.c_float * 3),
        ("mhr_model_params", ctypes.c_float * 204),
        ("skel_3d",         ctypes.c_float * 381),   # [127 × 3]
        ("has_skel",        ctypes.c_int),
    ]


def load_library(lib_dir: str) -> ctypes.CDLL:
    lib_path = os.path.join(lib_dir, "libfast_sam_3dbody.so")
    if not os.path.exists(lib_path):
        sys.exit(f"Library not found: {lib_path}\nBuild the project first.")

    prev    = os.environ.get("LD_LIBRARY_PATH", "")
    ort_lib = os.path.join(lib_dir, "onnxruntime_dl", "lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [lib_dir, ort_lib, prev]))

    lib = ctypes.CDLL(lib_path)
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
    return lib


# ──────────────────────────────────────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────────────────────────────────────

def result_to_smplx_dict(result: FsbResult) -> dict:
    """
    Convert an FsbResult to {dpose_joint_name: [x, y, z]} using the D-PoSE
    SMPL-X joint naming (body + hands + head).  Returns {} when no keypoints.
    """
    if not result.has_kps:
        return {}

    kps_3d = np.array(result.kps_3d, dtype=np.float32).reshape(70, 3)
    out = {}

    # Body joints renamed from MHR70.
    for dst, src in SMPLX_BODY_FROM_MHR70.items():
        out[dst] = kps_3d[MHR70_IDX[src]].tolist()

    # Pelvis + spine1/2/3 from the full 127-joint skeleton.
    if result.has_skel:
        skel = np.array(result.skel_3d, dtype=np.float32).reshape(127, 3)
        for dst, src in SMPLX_BODY_FROM_SKEL.items():
            out[dst] = skel[MHR127_IDX[src]].tolist()
    else:
        # Fallback when the skeleton is unavailable (e.g. skip_body_model):
        # pelvis = hip midpoint; spines linearly interpolated pelvis→neck.
        pelvis = (kps_3d[MHR70_IDX["left_hip"]] + kps_3d[MHR70_IDX["right_hip"]]) * 0.5
        neck   = kps_3d[MHR70_IDX["neck"]]
        out["pelvis"] = pelvis.tolist()
        out["spine1"] = (pelvis + 0.25 * (neck - pelvis)).tolist()
        out["spine2"] = (pelvis + 0.50 * (neck - pelvis)).tolist()
        out["spine3"] = (pelvis + 0.75 * (neck - pelvis)).tolist()

    # head (SMPL-X body index 15): approximate as ear midpoint.
    out["head"] = ((kps_3d[MHR70_IDX["left_ear"]] +
                    kps_3d[MHR70_IDX["right_ear"]]) * 0.5).tolist()

    # Hands.
    for dst, src in HAND_MAP.items():
        out[dst] = kps_3d[MHR70_IDX[src]].tolist()

    # Head landmarks.
    for dst, src in HEAD_MAP.items():
        out[dst] = kps_3d[MHR70_IDX[src]].tolist()

    return out


# ──────────────────────────────────────────────────────────────────────────────
# CSV export  (mirrors D-PoSE demo_webcam_csv.py write_skeleton_rows_to_csv)
# ──────────────────────────────────────────────────────────────────────────────

def _skeleton_dict_to_csv_row(skeleton_dict: dict, frame_id: int, skeleton_id: int) -> dict:
    row = {"frame_id": int(frame_id), "skeleton_id": int(skeleton_id)}
    for joint_name, coords in skeleton_dict.items():
        if coords is None:
            continue
        try:
            x, y, z = coords[:3]
        except Exception:
            continue
        row[f"{joint_name}_x"] = float(x)
        row[f"{joint_name}_y"] = float(y)
        row[f"{joint_name}_z"] = float(z)
    return row


def write_skeleton_rows_to_csv(rows: list, csv_path: str) -> None:
    import csv
    if rows is None:
        rows = []

    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fixed = ["frame_id", "skeleton_id"]
    other = sorted(k for k in all_keys if k not in fixed)
    fieldnames = fixed + other

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[INFO] Skeleton CSV saved to {csv_path}  ({len(rows)} row(s))")


# ──────────────────────────────────────────────────────────────────────────────
# Legacy CSV writers  (mirror D-PoSE tools.py, single skeleton-0 per frame)
# ──────────────────────────────────────────────────────────────────────────────

def get_azure_kinect_skeleton() -> list:
    """Joint order for the Azure-Kinect-style legacy CSV (D-PoSE tools.py)."""
    return [
        "pelvis", "spine1", "spine2", "neck",
        "left_collar", "left_shoulder", "left_elbow", "left_wrist",
        "left_hand_middle1", "left_hand_middle4", "left_hand_thumb4",
        "right_collar", "right_shoulder", "right_elbow", "right_wrist",
        "right_hand_middle1", "right_hand_middle4", "right_hand_thumb4",
        "left_hip", "left_knee", "left_ankle", "left_foot",
        "right_hip", "right_knee", "right_ankle", "right_foot",
        "head_jaw", "head", "head_left_eye", "head_left_ear",
        "head_right_eye", "head_right_ear",
    ]


def save_csv_list_of_dicts(filename: str, history: list) -> None:
    """Legacy 3DPoints.csv format: header <joint>_3DX,_3DY,_3DZ, one frame/row."""
    labels = []
    for frame in history:
        for label in frame:
            if label not in labels:
                labels.append(label)

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, "w") as f:
        for col, label in enumerate(labels):
            if col > 0:
                f.write(",")
            f.write(f"{label}_3DX,{label}_3DY,{label}_3DZ")
        f.write("\n")
        for frame in history:
            for col, label in enumerate(labels):
                if col > 0:
                    f.write(",")
                if label in frame:
                    x, y, z = frame[label][:3]
                    f.write(f"{x:f},{y:f},{z:f}")
                else:
                    f.write("0,0,0")
            f.write("\n")
    print(f"[INFO] Legacy CSV saved to {filename}  ({len(history)} frame(s))")


def save_csv_skeleton_order(filename: str, history: list, skeleton: list) -> None:
    """Legacy CSV with joint columns in a fixed skeleton order (missing → 0,0,0)."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, "w") as f:
        for i, joint in enumerate(skeleton):
            if i > 0:
                f.write(",")
            f.write(f"{joint}_3DX,{joint}_3DY,{joint}_3DZ")
        f.write("\n")
        for frame in history:
            for i, joint in enumerate(skeleton):
                if i > 0:
                    f.write(",")
                if joint in frame:
                    c = frame[joint]
                    f.write(f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f}")
                else:
                    f.write("0,0,0")
            f.write("\n")
    print(f"[INFO] Legacy skeleton-order CSV saved to {filename}  ({len(history)} frame(s))")


def save_json(obj, filename: str) -> None:
    import json
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)
    print(f"[INFO] JSON saved to {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Frame helpers  (mirror D-PoSE scale_and_embed_frame + a 2-D overlay)
# ──────────────────────────────────────────────────────────────────────────────

def scale_and_embed_frame(frame, target_w=1280, target_h=720):
    """Resize + letterbox-pad a frame into a fixed target_w×target_h canvas."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def draw_overlay(frame, results, n, hud):
    """Draw projected 2-D keypoints + bbox + HUD on a copy of the frame."""
    vis = frame.copy()
    for i in range(n):
        r = results[i]
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1, cv2.LINE_AA)
        if r.has_kps:
            kps = np.array(r.kps_2d, dtype=np.float32).reshape(70, 2)
            for px, py in kps:
                cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(vis, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def encode_video_from_jpgs(folder, framerate, width, height):
    """ffmpeg the saved colorFrame_0_*.jpg into livelastRun3DHiRes.mp4, then delete them."""
    import glob
    import subprocess
    out_mp4 = os.path.join(folder, "livelastRun3DHiRes.mp4")
    cmd = [
        "ffmpeg", "-framerate", str(framerate), "-start_number", "1",
        "-i", os.path.join(folder, "colorFrame_0_%05d.jpg"),
        "-s", f"{width}x{height}", "-y", "-r", str(framerate),
        "-pix_fmt", "yuv420p", "-threads", "8", out_mp4,
    ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for jpg in glob.glob(os.path.join(folder, "colorFrame_0_*.jpg")):
            os.remove(jpg)
        print(f"[INFO] Video saved to {out_mp4}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[WARN] ffmpeg encode failed ({e}); leaving jpgs in {folder}")


# ──────────────────────────────────────────────────────────────────────────────
# Capture helper (mirror D-PoSE getCaptureDeviceFromPath for /dev/videoX, files)
# ──────────────────────────────────────────────────────────────────────────────

def open_capture(src: str):
    """Returns (cap_or_None, is_image)."""
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    if any(src.lower().endswith(e) for e in img_exts):
        return None, True
    if src.isdigit():
        return cv2.VideoCapture(int(src)), False
    if src.startswith("/dev/video"):
        return cv2.VideoCapture(int(src.replace("/dev/video", "")) if
                                src.replace("/dev/video", "").isdigit() else src), False
    return cv2.VideoCapture(src), False


# ──────────────────────────────────────────────────────────────────────────────
# CLI  (argument names kept close to D-PoSE demo_webcam_csv.py)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    cpp_dir = os.path.dirname(os.path.abspath(__file__))
    onnx    = os.path.join(cpp_dir, "onnx")
    build   = os.path.join(cpp_dir, "build")

    p = argparse.ArgumentParser(
        description="SAM-3D-Body → D-PoSE-compatible SMPL-X CSV dumper")

    # D-PoSE-style arguments
    p.add_argument("--input", type=str, default="/dev/video0",
                   help="Input source: /dev/videoX, webcam index, video, or image path")
    p.add_argument("--output_folder", type=str, default="demo_images/results",
                   help="Folder where 3DPoints.csv is written")
    p.add_argument("--save", action=argparse.BooleanOptionalAction,
                   help="Also write per-frame CSV+JSON, save colorFrame jpgs and "
                        "ffmpeg them into livelastRun3DHiRes.mp4")
    p.add_argument("--display", action="store_true",
                   help="Show the video window while processing")
    p.add_argument("--no-letterbox", dest="letterbox", action="store_false",
                   help="Disable the D-PoSE 1280x720 letterbox; run at native size")
    p.add_argument("--width",     type=int, default=1280, help="Letterbox canvas width")
    p.add_argument("--height",    type=int, default=720,  help="Letterbox canvas height")
    p.add_argument("--framerate", type=int, default=30,   help="ffmpeg output fps for --save")

    # SAM-3D-Body specific paths / params
    p.add_argument("--lib-dir",       default=build)
    p.add_argument("--onnx-dir",      default=onnx)
    p.add_argument("--gguf",          default=os.path.join(onnx, "pipeline.gguf"))
    p.add_argument("--yolo",          default=os.path.join(onnx, "yolo.onnx"))
    p.add_argument("--cuda",          type=int, default=0)
    p.add_argument("--max-skeletons", type=int, default=0,
                   help="Max persons per frame (0 = unlimited)")
    p.add_argument("--thresh",        type=float, default=0.5)
    p.add_argument("--nms",           type=float, default=0.45)
    p.add_argument("--fx",            type=float, default=0.0)
    p.add_argument("--fy",            type=float, default=0.0)
    p.add_argument("--cx",            type=float, default=0.0)
    p.add_argument("--cy",            type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    lib    = load_library(args.lib_dir)
    handle = lib.fsb_create()
    if not handle:
        sys.exit("fsb_create() returned NULL")

    cfg = FsbConfig(
        onnx_dir         = args.onnx_dir.encode(),
        gguf_path        = args.gguf.encode(),
        yolo_path        = args.yolo.encode(),
        cuda_device      = args.cuda,
        skip_body_model  = 0,
        person_thresh    = args.thresh,
        person_nms_iou   = args.nms,
        max_persons      = args.max_skeletons,
        focal_x          = args.fx,
        focal_y          = args.fy,
        principal_x      = args.cx,
        principal_y      = args.cy,
        zero_face_params = 0,
        detector         = 0,
    )

    print("Loading pipeline …")
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        lib.fsb_destroy(handle)
        sys.exit("Pipeline load failed")
    print("Pipeline ready.\n")

    os.makedirs(args.output_folder, exist_ok=True)

    MAX_RESULTS = max(args.max_skeletons if args.max_skeletons > 0 else 32, 32)
    ResultArray = FsbResult * MAX_RESULTS
    results_buf = ResultArray()

    cap, is_image = open_capture(args.input)
    if not is_image and (cap is None or not cap.isOpened()):
        lib.fsb_destroy(handle)
        sys.exit(f"Cannot open input: {args.input}")

    skeleton_rows = []     # all (frame, skeleton) rows across the run
    history       = []     # skeleton-0 joint dict per frame (legacy CSV / JSON)
    frame_number  = 0
    fps_ema       = 0.0
    prev_t        = time.perf_counter()

    print("Recording … press Q to stop and save CSV.")

    while True:
        if is_image:
            frame = cv2.imread(args.input)
            if frame is None:
                sys.exit(f"Cannot read image: {args.input}")
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        frame_number += 1

        # D-PoSE-style letterbox to a fixed canvas before inference.
        if args.letterbox:
            frame = scale_and_embed_frame(frame, args.width, args.height)

        # fsb_process_bgr requires a contiguous buffer.
        frame   = np.ascontiguousarray(frame)
        H, W    = frame.shape[:2]
        bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        t_inf = time.perf_counter()
        n     = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, MAX_RESULTS)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        frame_rows = []
        for i in range(n):
            joint_dict = result_to_smplx_dict(results_buf[i])
            if joint_dict:
                row = _skeleton_dict_to_csv_row(joint_dict, frame_number, i)
                skeleton_rows.append(row)
                frame_rows.append(row)
                if i == 0:
                    history.append(joint_dict)

        now     = time.perf_counter()
        fps_ema = 0.1 / (now - prev_t) + 0.9 * fps_ema
        prev_t  = now

        hud = f"FPS {fps_ema:.1f} | {inf_ms:.0f} ms | {n} person(s) | frame {frame_number}"

        if args.save:
            if frame_rows:
                write_skeleton_rows_to_csv(
                    frame_rows, os.path.join(args.output_folder, "skeleton_%05u.csv" % frame_number))
                save_json(history[-1] if history else {},
                          os.path.join(args.output_folder, "skeleton_%05u.json" % frame_number))
            # colorFrame jpg (with overlay) for the output video.
            cv2.imwrite(os.path.join(args.output_folder, "colorFrame_0_%05d.jpg" % frame_number),
                        draw_overlay(frame, results_buf, n, hud))

        if args.display:
            cv2.imshow("SAM-3D-Body → D-PoSE CSV", draw_overlay(frame, results_buf, n, hud))
            if (cv2.waitKey(1 if not is_image else 0) & 0xFF) in (ord("q"), 27):
                break

        if is_image:
            break

    if cap:
        cap.release()
    if args.display:
        cv2.destroyAllWindows()
    lib.fsb_destroy(handle)

    if not skeleton_rows:
        print("No keypoints captured — no CSV written.")
        return

    out_csv = os.path.join(args.output_folder, "3DPoints.csv")
    write_skeleton_rows_to_csv(skeleton_rows, out_csv)

    # Legacy CSV variants + history JSON (mirror D-PoSE demo_webcam.py outputs).
    save_csv_list_of_dicts(os.path.join(args.output_folder, "3DPoints_legacy.csv"), history)
    save_csv_skeleton_order(os.path.join(args.output_folder, "3DPointsAzureKinect_legacy.csv"),
                            history, get_azure_kinect_skeleton())
    save_json(history, os.path.join(args.output_folder, "mhr_history.json"))

    # Encode the saved colorFrame jpgs into a video, then remove them.
    if args.save:
        encode_video_from_jpgs(args.output_folder, args.framerate, W, H)

    print(f"\nDone. {frame_number} frame(s) processed, {len(skeleton_rows)} skeleton row(s).")


if __name__ == "__main__":
    main()
