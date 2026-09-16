#!/usr/bin/env python3
"""
fast_sam_3dbody_frontend-3D.py
Heavy frontend: C engine for detection + backbone + decoder + MHR FFN heads,
Python MHR body model for full 3D mesh, rendered like demo_webcam.py.

Usage:
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_frontend-3D.py --from assets/teaser.png
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_frontend-3D.py --from 0 --max-skeletons 3
  python fast_sam_3dbody_cpp/python/fast_sam_3dbody_frontend-3D.py --from video.mp4 --headless --out out.mp4
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np
import torch

# Add repo root so sam_3d_body package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _repo_root)

from sam_3d_body.build_models import load_sam_3d_body
from tools.vis_utils import visualize_sample_together

# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs  (must match fast_sam_3dbody_capi.h exactly)
# ──────────────────────────────────────────────────────────────────────────────

from fsb_ctypes import FsbConfig, FsbResult, load_library


# ──────────────────────────────────────────────────────────────────────────────
# Per-person body model inference + packaging
# ──────────────────────────────────────────────────────────────────────────────

def _correct_pred_cam_t(pred_cam_t, j3d_np, result, fl, px, py,
                         conf_thresh=0.3):
    """
    Correct tx, ty in pred_cam_t using YOLO 2D keypoints and Python 3D joints.

    The C++ camera head single-pass output has unreliable tx, ty for
    non-frontal or unusual poses. YOLO gives accurate 2D positions for the
    17 COCO keypoints, and the first 17 MHR70 joints (j3d_np[:17]) are the
    same COCO keypoints in 3D body space. Together they linearly constrain:
        tx_k = (yolo_x[k] - px) * (j3d[k,2] + tz) / fl - j3d[k,0]
    Averaging over visible joints gives a robust corrected tx, ty.
    Falls back to a bbox-center estimate when no YOLO kps are available.
    """
    tz = float(pred_cam_t[2])

    # COCO index → MHR70 index. MHR70[9]=left_hip (not wrist!); wrists are at 62,41.
    _COCO_TO_MHR70 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]

    if result.has_yolo_kps:
        yolo = np.array(list(result.yolo_kps[:51])).reshape(17, 3)

        txs, tys = [], []
        for k in range(17):
            conf = float(yolo[k, 2])
            if conf < conf_thresh:
                continue
            j3d_k = j3d_np[_COCO_TO_MHR70[k]]
            d = float(j3d_k[2]) + tz
            if d < 1e-3:
                continue
            txs.append((float(yolo[k, 0]) - px) * d / fl - float(j3d_k[0]))
            tys.append((float(yolo[k, 1]) - py) * d / fl - float(j3d_k[1]))

        if txs:
            return np.array([np.mean(txs), np.mean(tys), tz], dtype=np.float32)

    # Fallback: project body root to bbox center
    bbox = np.array(list(result.bbox[:4]))
    bbox_cx = (bbox[0] + bbox[2]) * 0.5
    bbox_cy = (bbox[1] + bbox[3]) * 0.5
    root = j3d_np[0]  # pelvis ≈ [0,0,0] in body space
    d = float(root[2]) + tz
    tx = (bbox_cx - px) * d / fl - float(root[0])
    ty = (bbox_cy - py) * d / fl - float(root[1])
    return np.array([tx, ty, tz], dtype=np.float32)


def fsb_result_to_output(model, result: FsbResult, frame_h: int, frame_w: int,
                          device: str, principal_x: float, principal_y: float):
    """
    Run Python MHR body model on pose params from the C engine.
    Returns a dict compatible with visualize_sample_together.
    """
    def _t(arr, n):
        return torch.tensor(list(arr)[:n], dtype=torch.float32, device=device).unsqueeze(0)

    global_rot  = _t(result.global_rot,  3)
    body_pose   = _t(result.body_pose,  133)
    shape       = _t(result.shape,       45)
    scale       = _t(result.scale,       28)
    hand_pose   = _t(result.hand_pose,  108)
    face_params = _t(result.face_params, 72)

    # Python's training path always zeroes body_pose[62:116] (mhr_param_hand_idxs)
    # before replace_hands_in_pose fills the actual hand joints.  C++ compact_cont
    # fills those positions with values from the continuous representation that the
    # body model never sees during training — passing them through causes distorted
    # forearms and feet.  Mirror the Python convention by zeroing them here.
    body_pose[:, 62:116] = 0.0

    with torch.no_grad():
        out = model.head_pose.mhr_forward(
            global_trans=torch.zeros(1, 3, device=device),
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale,
            shape_params=shape,
            expr_params=face_params,
            return_keypoints=True,
        )

    if isinstance(out, tuple):
        verts, j3d = out[0], out[1]   # [1,18439,3], [1,308,3]
    else:
        verts, j3d = out, None

    # Match Python pipeline coordinate convention: [1,2] *= -1
    verts = verts.clone()
    verts[..., [1, 2]] *= -1

    if j3d is not None:
        j3d = j3d[:, :70].clone()   # [1, 70, 3]
        j3d[..., [1, 2]] *= -1

    verts_np = verts[0].cpu().float().numpy()   # [18439, 3]

    bbox = np.array(list(result.bbox[:4]))
    fl   = float(result.focal_length)
    px   = principal_x if principal_x > 0.0 else frame_w * 0.5
    py   = principal_y if principal_y > 0.0 else frame_h * 0.5

    # Correct tx,ty using YOLO 2D ↔ MHR70 3D COCO joint correspondences.
    # The C++ single-pass camera head gives unreliable tx,ty; YOLO 2D kps
    # (same 17 COCO joints as j3d_np[:17]) let us solve for them linearly.
    if j3d is not None:
        j3d_np = j3d[0].cpu().float().numpy()   # [70, 3]
        pred_cam_t = _correct_pred_cam_t(
            np.array(list(result.pred_cam_t[:3])), j3d_np, result, fl, px, py)
    else:
        pred_cam_t = np.array(list(result.pred_cam_t[:3]))
        j3d_np = None

    # Project 3-D keypoints to 2-D image space
    if j3d_np is not None:
        j3d_cam = j3d_np + pred_cam_t                    # [70, 3]
        dz = np.maximum(j3d_cam[:, 2:3], 1e-4)
        j2d = j3d_cam[:, :2] / dz * fl + np.array([px, py])  # [70, 2]
    else:
        j2d = np.zeros((70, 2), dtype=np.float32)

    return {
        "bbox":              bbox,
        "focal_length":      fl,
        "pred_cam_t":        pred_cam_t,
        "pred_vertices":     verts_np,
        "pred_keypoints_2d": j2d,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAM-3D-Body 3D frontend")
    cpp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx    = os.path.join(cpp_dir, "onnx")
    build   = os.path.join(cpp_dir, "build")
    ckpt_default = os.path.join(_repo_root, "checkpoints", "sam-3d-body-dinov3")

    p.add_argument("--lib-dir",     default=build)
    p.add_argument("--onnx-dir",    default=onnx)
    p.add_argument("--gguf",        default=os.path.join(onnx, "pipeline.gguf"))
    p.add_argument("--yolo",        default=os.path.join(onnx, "yolo.onnx"))
    p.add_argument("--checkpoint",  default=os.path.join(ckpt_default, "model.ckpt"),
                   help="Path to SAM-3D-Body model.ckpt")
    p.add_argument("--mhr-model",   default=os.path.join(ckpt_default, "assets", "mhr_model.pt"),
                   help="Path to mhr_model.pt (TorchScript MHR body model)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                   help="PyTorch device for body model (default: auto)")
    p.add_argument("--from",        dest="src", default="0",
                   help="Webcam index, image path, or video path")
    p.add_argument("--cuda",        type=int,   default=0,
                   help="CUDA device index for C engine (-1 = CPU)")
    p.add_argument("--max-skeletons", type=int, default=0,
                   help="Max persons to process (0 = unlimited)")
    p.add_argument("--thresh",      type=float, default=0.5)
    p.add_argument("--nms",         type=float, default=0.45)
    p.add_argument("--fx",          type=float, default=0.0)
    p.add_argument("--fy",          type=float, default=0.0)
    p.add_argument("--cx",          type=float, default=0.0)
    p.add_argument("--cy",          type=float, default=0.0)
    p.add_argument("--headless",    action="store_true")
    p.add_argument("--out",         default="",
                   help="Write output to file (image or video)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load Python body model ────────────────────────────────────────────────
    print(f"[*] Loading SAM-3D-Body Python model from {args.checkpoint} …")
    t0 = time.perf_counter()
    model, _ = load_sam_3d_body(
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_model,
        device=args.device,
    )
    model.eval()
    faces = model.head_pose.faces.cpu().numpy()
    print(f"    body model ready in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Load C engine ─────────────────────────────────────────────────────────
    lib    = load_library(args.lib_dir)
    handle = lib.fsb_create()
    if not handle:
        sys.exit("fsb_create() returned NULL")

    cfg = FsbConfig(
        onnx_dir        = args.onnx_dir.encode(),
        gguf_path       = args.gguf.encode(),
        yolo_path       = args.yolo.encode(),
        cuda_device     = args.cuda,
        skip_body_model = 1,   # Python body model used instead
        person_thresh   = args.thresh,
        person_nms_iou  = args.nms,
        max_persons     = args.max_skeletons,
        focal_x         = args.fx,
        focal_y         = args.fy,
        principal_x     = args.cx,
        principal_y     = args.cy,
    )
    print("[*] Loading C engine …")
    t0 = time.perf_counter()
    if not lib.fsb_load(handle, ctypes.byref(cfg)):
        lib.fsb_destroy(handle)
        sys.exit("C engine load failed")
    print(f"    C engine ready in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    MAX_RESULTS  = max(args.max_skeletons if args.max_skeletons > 0 else 32, 32)
    ResultArray  = FsbResult * MAX_RESULTS
    results_buf  = ResultArray()

    # ── Open input ────────────────────────────────────────────────────────────
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

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if args.out and not is_image:
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer  = cv2.VideoWriter(args.out, fourcc, fps_src, (w_src * 4, h_src))

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_count = 0
    fps_ema     = 0.0
    fps_alpha   = 0.1
    prev_t      = time.perf_counter()

    while True:
        if is_image:
            frame = cv2.imread(src)
            if frame is None:
                sys.exit(f"Cannot read image: {src}")
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

        H, W = frame.shape[:2]
        bgr_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # ── C engine inference ────────────────────────────────────────────────
        t_inf = time.perf_counter()
        n = lib.fsb_process_bgr(handle, bgr_ptr, W, H, results_buf, MAX_RESULTS)
        inf_c_ms = (time.perf_counter() - t_inf) * 1000

        # ── Python body model ─────────────────────────────────────────────────
        t_py = time.perf_counter()
        outputs = []
        for i in range(n):
            try:
                out = fsb_result_to_output(
                    model, results_buf[i], H, W,
                    args.device, args.cx, args.cy,
                )
                outputs.append(out)
            except Exception as e:
                print(f"  [WARN] person {i} body model error: {e}")
        inf_py_ms = (time.perf_counter() - t_py) * 1000

        # ── Visualise ─────────────────────────────────────────────────────────
        if outputs:
            vis = visualize_sample_together(frame, outputs, faces).astype(np.uint8)
        else:
            # No detections: show original image 4× wide (match layout)
            blank = np.zeros_like(frame)
            vis   = np.concatenate([frame, blank, blank, blank], axis=1)

        # ── HUD ───────────────────────────────────────────────────────────────
        now     = time.perf_counter()
        fps_ema = fps_alpha / (now - prev_t) + (1 - fps_alpha) * fps_ema
        prev_t  = now
        frame_count += 1

        hud = (f"FPS {fps_ema:.1f}  |  C:{inf_c_ms:.0f}ms  "
               f"Py:{inf_py_ms:.0f}ms  |  {n} person(s)")
        if args.max_skeletons:
            hud += f"  [max {args.max_skeletons}]"
        h_vis = vis.shape[0]
        info_h = max(int(h_vis * 0.05), 20)
        vis[0:info_h, 0:600] = (vis[0:info_h, 0:600] * 0.3).astype(np.uint8)
        cv2.putText(vis, hud, (10, max(info_h - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # ── Output ────────────────────────────────────────────────────────────
        if writer:
            writer.write(vis)
        if args.out and is_image:
            cv2.imwrite(args.out, vis)
            print(f"Saved {args.out}")

        if not args.headless:
            cv2.imshow("SAM-3D-Body  [orig | 2D skeleton | front mesh | side mesh]", vis)
            key = cv2.waitKey(1 if not is_image else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and args.out == "":
                save = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save, vis)
                print(f"  Saved {save}")

        if is_image:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if writer:
        writer.release()
    if cap:
        cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    lib.fsb_destroy(handle)

    if frame_count:
        print(f"Processed {frame_count} frame(s)  avg {fps_ema:.1f} fps")


if __name__ == "__main__":
    main()
