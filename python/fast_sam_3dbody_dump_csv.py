#!/usr/bin/env python3
"""
fast_sam_3dbody_dump_csv.py
Runs the C++ SAM-3D-Body pipeline and dumps per-frame 3D keypoints to CSV,
compatible with the format produced by DumpDataFromWebcamFromOtherProject.py.

The 70 MHR keypoints cover:
  body  (0-20):  nose, eyes, ears, shoulders, elbows, hips, knees, ankles, feet
  right hand (21-41): thumb→pinky fingers + wrist
  left  hand (42-62): thumb→pinky fingers + wrist
  extra (63-69): olecranon, cubital-fossa, acromion, neck

CSV format (one file per person):
  header:  joint_3DX,joint_3DY,joint_3DZ,...
  rows:    one frame per row, float values (0,0,0 when joint absent)

Usage:
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_dump_csv.py --from 0
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_dump_csv.py --from video.mp4 --csv-out pose.csv
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_dump_csv.py --from 0 --headless --csv-out pose.csv
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
    "nose",                        #  0
    "left_eye",                    #  1
    "right_eye",                   #  2
    "left_ear",                    #  3
    "right_ear",                   #  4
    "left_shoulder",               #  5
    "right_shoulder",              #  6
    "left_elbow",                  #  7
    "right_elbow",                 #  8
    "left_hip",                    #  9
    "right_hip",                   # 10
    "left_knee",                   # 11
    "right_knee",                  # 12
    "left_ankle",                  # 13
    "right_ankle",                 # 14
    "left_big_toe_tip",            # 15
    "left_small_toe_tip",          # 16
    "left_heel",                   # 17
    "right_big_toe_tip",           # 18
    "right_small_toe_tip",         # 19
    "right_heel",                  # 20
    "right_thumb_tip",             # 21
    "right_thumb_first_joint",     # 22
    "right_thumb_second_joint",    # 23
    "right_thumb_third_joint",     # 24
    "right_index_tip",             # 25
    "right_index_first_joint",     # 26
    "right_index_second_joint",    # 27
    "right_index_third_joint",     # 28
    "right_middle_tip",            # 29
    "right_middle_first_joint",    # 30
    "right_middle_second_joint",   # 31
    "right_middle_third_joint",    # 32
    "right_ring_tip",              # 33
    "right_ring_first_joint",      # 34
    "right_ring_second_joint",     # 35
    "right_ring_third_joint",      # 36
    "right_pinky_tip",             # 37
    "right_pinky_first_joint",     # 38
    "right_pinky_second_joint",    # 39
    "right_pinky_third_joint",     # 40
    "right_wrist",                 # 41
    "left_thumb_tip",              # 42
    "left_thumb_first_joint",      # 43
    "left_thumb_second_joint",     # 44
    "left_thumb_third_joint",      # 45
    "left_index_tip",              # 46
    "left_index_first_joint",      # 47
    "left_index_second_joint",     # 48
    "left_index_third_joint",      # 49
    "left_middle_tip",             # 50
    "left_middle_first_joint",     # 51
    "left_middle_second_joint",    # 52
    "left_middle_third_joint",     # 53
    "left_ring_tip",               # 54
    "left_ring_first_joint",       # 55
    "left_ring_second_joint",      # 56
    "left_ring_third_joint",       # 57
    "left_pinky_tip",              # 58
    "left_pinky_first_joint",      # 59
    "left_pinky_second_joint",     # 60
    "left_pinky_third_joint",      # 61
    "left_wrist",                  # 62
    "left_olecranon",              # 63
    "right_olecranon",             # 64
    "left_cubital_fossa",          # 65
    "right_cubital_fossa",         # 66
    "left_acromion",               # 67
    "right_acromion",              # 68
    "neck",                        # 69
]

# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs  (must match fast_sam_3dbody_capi.h exactly)
# ──────────────────────────────────────────────────────────────────────────────

from fsb_ctypes import FsbConfig, FsbResult, load_library


# ──────────────────────────────────────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────────────────────────────────────

def result_to_joint_dict(result: FsbResult) -> dict:
    """
    Convert an FsbResult to {joint_name: [x, y, z]} for all 70 MHR keypoints.
    Returns empty dict when no keypoints are available.
    """
    if not result.has_kps:
        return {}
    kps_3d = np.array(result.kps_3d, dtype=np.float32).reshape(70, 3)
    return {MHR70_NAMES[i]: kps_3d[i].tolist() for i in range(70)}


def save_csv(filename: str, history: list) -> None:
    """
    Save a list of per-frame joint dicts to CSV.
    Matches the format of saveCSVFileFromListOfDicts in D-PoSE/tools.py:
      header:  joint_3DX,joint_3DY,joint_3DZ,...
      rows:    one frame per line, 0,0,0 when joint absent for that frame
    """
    labels = []
    for frame in history:
        for label in frame:
            if label not in labels:
                labels.append(label)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

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
                    x, y, z = frame[label]
                    f.write(f"{x:f},{y:f},{z:f}")
                else:
                    f.write("0,0,0")
            f.write("\n")

    print(f"Saved {len(history)} frame(s) → {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAM-3D-Body CSV keypoint dumper")
    cpp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx    = os.path.join(cpp_dir, "onnx")
    build   = os.path.join(cpp_dir, "build")

    p.add_argument("--lib-dir",       default=build)
    p.add_argument("--onnx-dir",      default=onnx)
    p.add_argument("--gguf",          default=os.path.join(onnx, "pipeline.gguf"))
    p.add_argument("--yolo",          default=os.path.join(onnx, "yolo.onnx"))
    p.add_argument("--from",          dest="src", default="0",
                   help="Webcam index, image path, or video path")
    p.add_argument("--cuda",          type=int, default=0)
    p.add_argument("--max-skeletons", type=int, default=1,
                   help="Max persons per frame (0 = unlimited, one CSV per person)")
    p.add_argument("--thresh",        type=float, default=0.5)
    p.add_argument("--nms",           type=float, default=0.45)
    p.add_argument("--fx",            type=float, default=0.0)
    p.add_argument("--fy",            type=float, default=0.0)
    p.add_argument("--cx",            type=float, default=0.0)
    p.add_argument("--cy",            type=float, default=0.0)
    p.add_argument("--headless",      action="store_true",
                   help="Do not open a display window")
    p.add_argument("--csv-out",       default="3DPoints.csv",
                   help="Output CSV path.  With multiple persons: person0_3DPoints.csv, …")
    return p.parse_args()


def main():
    args = parse_args()

    lib    = load_library(args.lib_dir)
    handle = lib.fsb_create()
    if not handle:
        sys.exit("fsb_create() returned NULL")

    cfg = FsbConfig(
        onnx_dir        = args.onnx_dir.encode(),
        gguf_path       = args.gguf.encode(),
        yolo_path       = args.yolo.encode(),
        cuda_device     = args.cuda,
        skip_body_model = 0,
        person_thresh   = args.thresh,
        person_nms_iou  = args.nms,
        max_persons     = args.max_skeletons,
        focal_x         = args.fx,
        focal_y         = args.fy,
        principal_x     = args.cx,
        principal_y     = args.cy,
    )

    print("Loading pipeline …")
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        lib.fsb_destroy(handle)
        sys.exit("Pipeline load failed")
    print("Pipeline ready.\n")

    MAX_RESULTS = max(args.max_skeletons if args.max_skeletons > 0 else 32, 32)
    ResultArray = FsbResult * MAX_RESULTS
    results_buf = ResultArray()

    src      = args.src
    is_image = False
    cap      = None

    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        if any(src.lower().endswith(e) for e in img_exts):
            is_image = True
        else:
            cap = cv2.VideoCapture(src)

    if not is_image and (cap is None or not cap.isOpened()):
        lib.fsb_destroy(handle)
        sys.exit(f"Cannot open input: {src}")

    # history[person_idx] = list of per-frame joint dicts
    history: dict[int, list] = {}
    frame_count = 0
    fps_ema     = 0.0
    fps_alpha   = 0.1
    prev_t      = time.perf_counter()

    print("Recording … press Q to stop and save CSV.")

    while True:
        if is_image:
            frame = cv2.imread(src)
            if frame is None:
                sys.exit(f"Cannot read image: {src}")
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        H, W    = frame.shape[:2]
        bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        t_inf  = time.perf_counter()
        n      = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, MAX_RESULTS)
        inf_ms = (time.perf_counter() - t_inf) * 1000
        frame_count += 1

        for i in range(n):
            joint_dict = result_to_joint_dict(results_buf[i])
            if joint_dict:
                history.setdefault(i, []).append(joint_dict)

        now     = time.perf_counter()
        fps_ema = fps_alpha / (now - prev_t) + (1 - fps_alpha) * fps_ema
        prev_t  = now

        if not args.headless:
            vis = frame.copy()
            hud = (f"FPS {fps_ema:.1f}  |  {inf_ms:.0f} ms  |  "
                   f"{n} person(s)  |  {frame_count} frames recorded")
            cv2.putText(vis, hud, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("SAM-3D-Body CSV Dump", vis)
            key = cv2.waitKey(1 if not is_image else 0) & 0xFF
            if key in (ord("q"), 27):
                break

        if is_image:
            break

    if cap:
        cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    lib.fsb_destroy(handle)

    if not history:
        print("No keypoints captured — no CSV written.")
        return

    base, ext = os.path.splitext(args.csv_out)
    for person_idx, frames in sorted(history.items()):
        out_path = args.csv_out if len(history) == 1 else f"{base}_person{person_idx}{ext}"
        save_csv(out_path, frames)

    print(f"\nDone. {frame_count} frame(s) processed, {len(history)} person track(s) saved.")


if __name__ == "__main__":
    main()
