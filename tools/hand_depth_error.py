#!/usr/bin/env python3
"""Absolute depth error between our C++ render and Python's real SAM-3D-Body
render, cropped to the hand regions.

Compares:
  - a raw float32 depth dump from `fast_sam_3dbody_render --save-depth`
    (row-major, W*H floats, metres, 0 = background/untouched pixel)
  - Python's real depth (.npy, same convention -- see
    scratchpad's render_python_gt.py, which uses pyrender with the SAME
    fx=fy=focal_length, cx=W/2, cy=H/2 camera as mhr_camera_matrices())

Both must be rendered from the SAME camera (same image, same forced bbox) for
the comparison to be meaningful -- this does not re-align anything itself.

Usage:
  python3 tools/hand_depth_error.py \\
      --ours /path/depth00001.bin --width 1080 --height 1382 \\
      --python /path/python_gt_depth.npy \\
      --crop 250 350 550 650 --crop 500 580 900 900 \\
      [--crop-names right_hand left_hand] [--save-diff-prefix /tmp/diff]

Each --crop is "x1 y1 x2 y2" in pixel coordinates. Reports, per crop, over
pixels where BOTH maps are non-background: mean/median/max absolute error,
RMSE, and how many pixels were valid in each map (a large mismatch in valid
pixel *count* between the two maps -- not just depth value -- is itself a
sign of a silhouette/pose mismatch, not just a depth-precision gap).
"""
import argparse
import numpy as np


def load_ours(path, w, h):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != w * h:
        raise ValueError(f"{path}: expected {w*h} floats ({w}x{h}), got {arr.size}")
    return arr.reshape(h, w)


def load_python(path):
    return np.load(path)


def report_crop(ours, py, box, name):
    x1, y1, x2, y2 = box
    a = ours[y1:y2, x1:x2]
    b = py[y1:y2, x1:x2]
    if a.shape != b.shape:
        # Python depth map may be a different resolution; resize the crop
        # bounds proportionally is out of scope -- require matching sizes.
        raise ValueError(f"{name}: shape mismatch {a.shape} vs {b.shape} -- "
                          f"are ours/python the same image size?")
    valid_a = a > 0
    valid_b = b > 0
    both = valid_a & valid_b
    n_total = a.size
    print(f"\n=== {name}  box=({x1},{y1},{x2},{y2})  {a.shape[1]}x{a.shape[0]}px ===")
    print(f"  valid in ours:   {valid_a.sum():6d}/{n_total} ({100*valid_a.sum()/n_total:.1f}%)")
    print(f"  valid in python: {valid_b.sum():6d}/{n_total} ({100*valid_b.sum()/n_total:.1f}%)")
    print(f"  valid in both:   {both.sum():6d}/{n_total} ({100*both.sum()/n_total:.1f}%)")
    if not both.any():
        print("  (no overlapping valid pixels -- cannot compute depth error)")
        return None
    diff = np.abs(a[both] - b[both])
    stats = {
        "mean_abs_err_m": float(diff.mean()),
        "median_abs_err_m": float(np.median(diff)),
        "max_abs_err_m": float(diff.max()),
        "rmse_m": float(np.sqrt((diff ** 2).mean())),
        "n_valid_both": int(both.sum()),
        "iou_valid_mask": float(both.sum() / (valid_a | valid_b).sum()),
    }
    for k, v in stats.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    return diff, both


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="raw float32 depth dump from --save-depth")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--python", required=True, help="Python's real depth .npy")
    ap.add_argument("--crop", nargs=4, type=int, action="append", required=True,
                     metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--crop-names", nargs="*", default=None)
    ap.add_argument("--save-diff-prefix", default=None,
                     help="if set, saves a per-crop |diff| heatmap PNG at <prefix>_<name>.png")
    args = ap.parse_args()

    ours = load_ours(args.ours, args.width, args.height)
    py = load_python(args.python)
    if py.shape != ours.shape:
        print(f"WARNING: python depth shape {py.shape} != ours {ours.shape} "
              f"-- crops must still be given in ours' pixel coordinates; "
              f"results may be misaligned if the two weren't rendered at "
              f"the same resolution.")

    names = args.crop_names or [f"crop{i}" for i in range(len(args.crop))]
    if len(names) != len(args.crop):
        raise ValueError("--crop-names must have exactly one name per --crop")

    for box, name in zip(args.crop, names):
        result = report_crop(ours, py, tuple(box), name)
        if result is None:
            continue
        diff, both = result
        if args.save_diff_prefix:
            import cv2
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            vis = np.zeros((h, w), dtype=np.uint8)
            full_diff = np.zeros((h, w), dtype=np.float32)
            full_diff[both] = diff
            dmax = diff.max() if diff.size else 1.0
            vis = (255 * np.clip(full_diff / max(dmax, 1e-6), 0, 1)).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis_color[~both] = 0
            path = f"{args.save_diff_prefix}_{name}.png"
            cv2.imwrite(path, vis_color)
            print(f"  saved diff heatmap: {path}")


if __name__ == "__main__":
    main()
