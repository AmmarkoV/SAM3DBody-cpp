#!/usr/bin/env python3
"""
football_tracker.py
Given the World cup 2026 :P
R&D experiment/tool: retrieve 3-D body poses for small players in football footage.

The SAM-3D-Body C++ pipeline (libfast_sam_3dbody.so) detects people with an
internal YOLO and only produces good poses when a person fills a decent part of
the input.  In football wide shots players are tiny (~25x50 px), so running the
pipeline on the full frame yields nothing usable.

This tool sidesteps that: for every bounding box in a detections JSON it crops
the player out (with context padding), UPSCALES the crop so the player is a
healthy ~256 px tall, runs the normal pipeline on that crop, then maps the
returned 2-D keypoints back to full-frame coordinates.

It is deliberately self-contained but reuses the ctypes bindings and drawing
helpers from the repo's fast_sam_3dbody_frontend.py (no C++ changes needed).

Usage:
  tools/.venv/bin/python tools/football_tracker.py \
      --video clean_sample_1000f_taccam.mp4 \
      --detections detections_1000f_taccam.json \
      --out football_poses.mp4 --json football_poses.json \
      --max-frames 200

Notes:
  * 3-D translation (pred_cam_t) is crop-relative and NOT in full-frame metric
    space — fine for R&D / 2-D overlay; ignore it for absolute positioning.
  * One pipeline call per box per frame, so it is slow.  Use --stride /
    --max-frames / --max-persons to keep experiments quick.
"""

import argparse
import ctypes
import json
import os
import sys
import time

import cv2
import numpy as np

# ── Reuse the existing C++ ctypes bindings + drawing helpers ──────────────────
# fast_sam_3dbody_frontend.py lives at the repo root (one level up from tools/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from fast_sam_3dbody_frontend import (  # noqa: E402
    FsbConfig, FsbResult, load_library,
    draw_mhr70, _correct_kps2d,
)


# ──────────────────────────────────────────────────────────────────────────────
# Detection JSON
# ──────────────────────────────────────────────────────────────────────────────

def load_detections(path: str) -> tuple[dict, dict]:
    """Return (video_meta, {frame_index: [ {track_id, bbox xyxy, conf}, ... ]})."""
    with open(path) as f:
        data = json.load(f)

    by_frame: dict[int, list] = {}
    for ann in data.get("annotations", []):
        fi = int(ann["frame_index"])
        objs = []
        for o in ann.get("objects", []):
            if o.get("category", "person") != "person":
                continue
            x, y, w, h = (float(v) for v in o["bbox"])           # xywh
            objs.append({
                "track_id": int(o.get("track_id", -1)),
                "xyxy": (x, y, x + w, y + h),
                "conf": float(o.get("confidence", 0.0)),
            })
        by_frame[fi] = objs
    return data.get("video", {}), by_frame


# ──────────────────────────────────────────────────────────────────────────────
# Per-player crop → upscale → pipeline → map back
# ──────────────────────────────────────────────────────────────────────────────

def make_crop(frame: np.ndarray, xyxy, pad: float, target_h: int,
              max_scale: float):
    """
    Crop the padded bbox region and upscale it so the player is ~target_h tall.

    Returns (crop_bgr, ox, oy, scale) where (ox, oy) is the crop origin in the
    full frame and scale is the upscale factor applied (crop_px = full_px*scale).
    Returns None when the box is degenerate.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    # Pad for context — the body model needs to see the whole figure plus margin.
    px, py = bw * pad, bh * pad
    cx1 = int(max(0, np.floor(x1 - px)))
    cy1 = int(max(0, np.floor(y1 - py)))
    cx2 = int(min(W, np.ceil(x2 + px)))
    cy2 = int(min(H, np.ceil(y2 + py)))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    scale = float(np.clip(target_h / bh, 1.0, max_scale))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    return crop, cx1, cy1, scale


def pick_person(results: list, n: int, target_cxy) -> int:
    """Pick the pipeline result whose bbox centre is closest to the target
    player's centre (the crop may contain partial neighbours).  Returns -1."""
    best, best_d = -1, 1e18
    tx, ty = target_cxy
    for i in range(n):
        bx = results[i].bbox
        cx = 0.5 * (bx[0] + bx[2])
        cy = 0.5 * (bx[1] + bx[3])
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        if d < best_d:
            best_d, best = d, i
    return best


_PERSON_COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 80, 0), (255, 0, 200),
    (200, 255, 0), (0, 100, 255), (180, 180, 255), (255, 200, 0),
]


def run(args):
    lib = load_library(args.lib_dir)

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
        person_nms_iou   = 0.45,
        max_persons      = 8,
        focal_x          = 0.0, focal_y = 0.0,
        principal_x      = 0.0, principal_y = 0.0,
        zero_face_params = 1,
        detector         = 0,
    )
    print("Loading pipeline …")
    t0 = time.perf_counter()
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        lib.fsb_destroy(handle)
        sys.exit("Pipeline load failed")
    print(f"Pipeline ready in {(time.perf_counter()-t0)*1000:.0f} ms")

    MAX_RESULTS = 8
    results_buf = (FsbResult * MAX_RESULTS)()

    _video_meta, dets = load_detections(args.detections)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        lib.fsb_destroy(handle)
        sys.exit(f"Cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (W, H))

    json_frames = []
    frame_idx = -1
    processed = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1

        if args.max_frames and processed >= args.max_frames:
            break
        if frame_idx % args.stride != 0:
            continue

        objs = dets.get(frame_idx, [])
        # Process the most confident players first when capped.
        objs = sorted(objs, key=lambda o: -o["conf"])
        if args.max_persons:
            objs = objs[:args.max_persons]

        vis = frame.copy() if writer is not None else None
        frame_record = {"frame_index": frame_idx, "players": []}

        for pi, ob in enumerate(objs):
            made = make_crop(frame, ob["xyxy"], args.pad,
                             args.target_height, args.max_scale)
            rec = {"track_id": ob["track_id"], "bbox_xyxy": [round(v, 1) for v in ob["xyxy"]],
                   "found": False}

            if made is not None:
                crop, ox, oy, scale = made
                ch, cw = crop.shape[:2]
                bgr_ptr = np.ascontiguousarray(crop).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint8))
                n = lib.fsb_process_bgr(handle, bgr_ptr, cw, ch,
                                        results_buf, MAX_RESULTS)
                if n > 0:
                    # Target player centre expressed in upscaled-crop pixels.
                    x1, y1, x2, y2 = ob["xyxy"]
                    tcx = (0.5 * (x1 + x2) - ox) * scale
                    tcy = (0.5 * (y1 + y2) - oy) * scale
                    sel = pick_person(results_buf, n, (tcx, tcy))
                    r = results_buf[sel]

                    # 70x2 keypoints in crop space (YOLO-corrected when possible).
                    kps_crop = _correct_kps2d(r, ch, cw)
                    if kps_crop is None and r.has_kps:
                        kps_crop = np.array(r.kps_2d, np.float32).reshape(70, 2)

                    if kps_crop is not None:
                        kps_full = kps_crop / scale + np.array([ox, oy], np.float32)
                        rec["found"] = True
                        rec["global_rot"] = [round(v, 4) for v in r.global_rot]
                        rec["body_pose"] = [round(v, 4) for v in r.body_pose]
                        rec["kps_2d"] = np.round(kps_full, 1).tolist()
                        if r.has_kps:
                            rec["kps_3d"] = np.round(
                                np.array(r.kps_3d, np.float32).reshape(70, 3), 4).tolist()

                        if vis is not None:
                            col = _PERSON_COLORS[pi % len(_PERSON_COLORS)]
                            draw_mhr70(vis, r, kp_radius=2, edge_thick=1,
                                       kps_override=kps_full)
                            cv2.rectangle(vis, (int(x1), int(y1)),
                                          (int(x2), int(y2)), col, 1)
                            cv2.putText(vis, f"#{ob['track_id']}",
                                        (int(x1), int(y1) - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                                        cv2.LINE_AA)

            if vis is not None and not rec["found"]:
                x1, y1, x2, y2 = (int(v) for v in ob["xyxy"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 60), 1)

            frame_record["players"].append(rec)

        json_frames.append(frame_record)
        processed += 1

        if vis is not None:
            found = sum(1 for p in frame_record["players"] if p["found"])
            cv2.putText(vis, f"frame {frame_idx}  {found}/{len(objs)} poses",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)

        if processed % 10 == 0:
            rate = processed / (time.perf_counter() - t_start)
            print(f"  frame {frame_idx}  ({processed} done, {rate:.1f} fps)")

    if writer:
        writer.release()
    cap.release()
    lib.fsb_destroy(handle)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"video": args.video, "fps": src_fps,
                       "width": W, "height": H, "frames": json_frames}, f)
        print(f"Wrote poses → {args.json}")
    if args.out:
        print(f"Wrote overlay → {args.out}")

    total = sum(len(fr["players"]) for fr in json_frames)
    hits = sum(1 for fr in json_frames for p in fr["players"] if p["found"])
    print(f"Done: {processed} frames, {hits}/{total} player poses recovered "
          f"({100*hits/max(total,1):.0f}%).")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    onnx = os.path.join(_REPO_ROOT, "onnx")
    build = os.path.join(_REPO_ROOT, "build")

    p.add_argument("--video", required=True, help="Input football video")
    p.add_argument("--detections", required=True,
                   help="Per-frame bbox JSON (xywh, like detections_*.json)")
    p.add_argument("--out", default="", help="Annotated mp4 output (optional)")
    p.add_argument("--json", default="", help="Pose JSON output (optional)")

    p.add_argument("--target-height", type=int, default=256,
                   help="Upscale each player crop to ~this many px tall")
    p.add_argument("--max-scale", type=float, default=12.0,
                   help="Cap on the upscale factor")
    p.add_argument("--pad", type=float, default=0.6,
                   help="Context padding as a fraction of bbox size per side")
    p.add_argument("--thresh", type=float, default=0.3,
                   help="Person detection threshold inside the crop")

    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--max-persons", type=int, default=0,
                   help="Cap players per frame (most confident first, 0 = all)")

    p.add_argument("--lib-dir", default=build)
    p.add_argument("--onnx-dir", default=onnx)
    p.add_argument("--gguf", default=os.path.join(onnx, "pipeline.gguf"))
    p.add_argument("--yolo", default=os.path.join(onnx, "yolo.onnx"))
    p.add_argument("--cuda", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
