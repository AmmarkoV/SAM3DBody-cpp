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
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

# ── Reuse the existing C++ ctypes bindings + drawing helpers ──────────────────
# fast_sam_3dbody_frontend.py lives at the repo root (one level up from tools/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from fast_sam_3dbody_frontend import (  # noqa: E402
    FsbConfig, FsbResult, load_library, _correct_kps2d,
    MHR70_EDGES, _mhr_color, _MHR_HEAD_JOINTS,
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


def color_for_id(track_id: int) -> tuple:
    """Stable, distinct BGR colour per track id (golden-ratio hue spacing) so a
    player keeps the same colour across frames."""
    hue = (int(track_id) * 0.61803398875) % 1.0          # 0..1, well spread
    hsv = np.uint8([[[int(hue * 179), 220, 255]]])        # OpenCV hue is 0..179
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class _Mp4Writer:
    """Web-friendly H.264 / yuv420p mp4 via an ffmpeg pipe (with +faststart so
    it streams from a webserver).  Falls back to OpenCV's mp4v when ffmpeg is
    not on PATH — that file plays in VLC but often not in browsers."""

    def __init__(self, path: str, w: int, h: int, fps: float):
        self.proc = None
        self.cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
                   # trunc(...) forces even dims, which yuv420p requires.
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found — falling back to OpenCV mp4v "
                  "(may not play in browsers).")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv = cv2.VideoWriter(path, fourcc, fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv is not None:
            self.cv.release()


def draw_skeleton_kps(frame: np.ndarray, kps: np.ndarray,
                      kp_radius: int = 2, edge_thick: int = 1) -> None:
    """Draw the 70-joint MHR70 skeleton from a bare 70x2 array (no FsbResult).
    Mirrors fast_sam_3dbody_frontend.draw_mhr70 so real and filled poses look
    identical."""
    H, W = frame.shape[:2]
    for i, j in MHR70_EDGES:
        p1 = (int(kps[i, 0]), int(kps[i, 1]))
        p2 = (int(kps[j, 0]), int(kps[j, 1]))
        if not (0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H):
            continue
        head = i in _MHR_HEAD_JOINTS or j in _MHR_HEAD_JOINTS
        cv2.line(frame, p1, p2, _mhr_color(i),
                 edge_thick * 2 if head else edge_thick, cv2.LINE_AA)
    for k in range(70):
        pt = (int(kps[k, 0]), int(kps[k, 1]))
        if not (0 <= pt[0] < W and 0 <= pt[1] < H):
            continue
        r = kp_radius * 2 if k in _MHR_HEAD_JOINTS else kp_radius
        cv2.circle(frame, pt, r, _mhr_color(k), -1, cv2.LINE_AA)


def _bbox_anchor(bbox):
    """Return (centre[2], height) of an xyxy box — the position + scale used to
    normalise a pose so it is independent of where/how big the player is."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], np.float32), max(y2 - y1, 1.0)


def fill_track_gaps(json_frames: list, max_gap: int) -> int:
    """Fill kps_2d for rejected/failed frames by interpolating each track's body
    pose from its surrounding good frames.

    Position and scale always come from the current frame's detection box (which
    is reliable every frame on a fixed camera); only the body articulation is
    carried/interpolated.  Gaps longer than max_gap frames are left empty rather
    than invented.  Returns the number of poses filled.
    """
    seq: dict[int, list] = defaultdict(list)         # track_id -> [(frame_idx, rec), ...]
    for fr in json_frames:
        for p in fr["players"]:
            seq[p["track_id"]].append((fr["frame_index"], p))

    filled = 0
    for recs in seq.values():
        # normalised pose for each good frame (centre-subtracted, height-scaled)
        good = {}
        for k, (_, p) in enumerate(recs):
            if p.get("found") and "kps_2d" in p:
                c, s = _bbox_anchor(p["bbox_xyxy"])
                good[k] = (np.array(p["kps_2d"], np.float32) - c) / s
        if not good:
            continue
        gidx = sorted(good)

        for k, (fi, p) in enumerate(recs):
            if k in good or "kps_2d" in p:
                continue
            prev = max((g for g in gidx if g < k), default=None)
            nxt  = min((g for g in gidx if g > k), default=None)

            if prev is not None and nxt is not None:
                fp, fn = recs[prev][0], recs[nxt][0]
                if fn - fp > max_gap:
                    continue                          # gap too long → don't invent
                a = (fi - fp) / (fn - fp)
                norm = (1 - a) * good[prev] + a * good[nxt]
            elif prev is not None:
                if fi - recs[prev][0] > max_gap:
                    continue
                norm = good[prev]                     # hold (no future sample)
            else:
                if recs[nxt][0] - fi > max_gap:
                    continue
                norm = good[nxt]                      # hold backwards (track start)

            c, s = _bbox_anchor(p["bbox_xyxy"])
            p["kps_2d"] = np.round(norm * s + c, 1).tolist()
            p["filled"] = True
            filled += 1
    return filled


def render_overlay(video: str, out: str, json_frames: list,
                   W: int, H: int, fps: float) -> None:
    """Second pass: re-decode the video and draw poses (real + filled) from the
    collected/filled records."""
    by_idx = {fr["frame_index"]: fr for fr in json_frames}
    if not by_idx:
        return
    last = max(by_idx)

    cap = cv2.VideoCapture(video)
    writer = _Mp4Writer(out, W, H, fps)
    fidx = -1
    while True:
        ok, vis = cap.read()
        if not ok or vis is None:
            break
        fidx += 1
        if fidx > last:
            break
        fr = by_idx.get(fidx)
        if fr is None:                                # frame skipped by --stride
            continue

        n_real = 0
        for p in fr["players"]:
            col = color_for_id(p["track_id"])
            x1, y1, x2, y2 = (int(v) for v in p["bbox_xyxy"])
            if "kps_2d" in p:
                draw_skeleton_kps(vis, np.asarray(p["kps_2d"], np.float32))
                filled = p.get("filled", False)
                n_real += not filled
                box_col = tuple(c // 2 for c in col) if filled else col
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_col, 1)
                label = f"#{p['track_id']}" + ("~" if filled else "")
                cv2.putText(vis, label, (x1, max(y1 - 3, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_col, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 60), 1)

        cv2.putText(vis, f"frame {fidx}  {n_real}/{len(fr['players'])} poses",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)

    writer.close()
    cap.release()


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

    # ── Phase 1: inference (per crop) — collect records, no drawing yet ───────
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

        frame_record = {"frame_index": frame_idx, "players": []}

        for ob in objs:
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

                    # Size sanity vs the detection box.  Camera FOV/distance is
                    # ~constant across the match, so a player's pixel height is
                    # bounded by its detection box.  When the depth solve in
                    # _correct_kps2d goes bad the whole skeleton scales up
                    # uniformly — reject those by comparing the core-joint height
                    # (nose→ankles, robust to a single flung hand) to the box.
                    if kps_crop is not None and args.max_size_ratio > 0:
                        core_h = float(kps_full[0:21, 1].max()
                                       - kps_full[0:21, 1].min())
                        if (y2 - y1) > 1 and \
                           core_h > args.max_size_ratio * (y2 - y1):
                            rec["reject"] = "oversize"
                            kps_crop = None    # treat like a non-detection

                    if kps_crop is not None:
                        rec["found"] = True
                        rec["global_rot"] = [round(v, 4) for v in r.global_rot]
                        rec["body_pose"] = [round(v, 4) for v in r.body_pose]
                        rec["kps_2d"] = np.round(kps_full, 1).tolist()
                        if r.has_kps:
                            rec["kps_3d"] = np.round(
                                np.array(r.kps_3d, np.float32).reshape(70, 3), 4).tolist()

            frame_record["players"].append(rec)

        json_frames.append(frame_record)
        processed += 1

        if processed % 10 == 0:
            rate = processed / (time.perf_counter() - t_start)
            print(f"  frame {frame_idx}  ({processed} done, {rate:.1f} fps)")

    cap.release()
    lib.fsb_destroy(handle)

    # ── Phase 2: temporal fill across rejected / failed frames ────────────────
    n_filled = 0
    if args.temporal_fill:
        n_filled = fill_track_gaps(json_frames, args.fill_max_gap)
        print(f"Temporal fill: {n_filled} pose(s) interpolated across gaps "
              f"(max gap {args.fill_max_gap} frames)")

    # ── Phase 3: render overlay (real + filled poses) ─────────────────────────
    if args.out:
        render_overlay(args.video, args.out, json_frames, W, H, src_fps)

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
          f"({100*hits/max(total,1):.0f}%), {n_filled} filled by temporal pass.")


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
    p.add_argument("--max-size-ratio", type=float, default=2.0,
                   help="Reject a pose when its skeleton height exceeds this "
                        "multiple of the detection bbox height (fixed-camera "
                        "sanity filter; 0 = disabled)")
    p.add_argument("--temporal-fill", dest="temporal_fill", action="store_true",
                   default=True,
                   help="Interpolate each track's pose across rejected/failed "
                        "frames from neighbouring good frames (default: on)")
    p.add_argument("--no-temporal-fill", dest="temporal_fill",
                   action="store_false",
                   help="Disable temporal gap filling")
    p.add_argument("--fill-max-gap", type=int, default=10,
                   help="Longest gap (frames) to bridge per track when filling")

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
