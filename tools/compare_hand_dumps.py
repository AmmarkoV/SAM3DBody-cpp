#!/usr/bin/env python3
"""Compare C++ hand dumps against the official sam-3d-body ground truth
(dumped by tools/dump_official_hand_gt.py into /tmp/official_hand_gt).

Usage: python3 tools/compare_hand_dumps.py [--cpp-dir /tmp/cpp_run/dump]
"""
import os
import sys

import numpy as np

GT = "/tmp/official_hand_gt"
CPP = sys.argv[sys.argv.index("--cpp-dir") + 1] if "--cpp-dir" in sys.argv else "/tmp/cpp_run/dump"


def load_npy(name):
    return np.load(os.path.join(GT, name + ".npy"))


def load_txt(path, cols=None):
    vals = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            vals.append([float(p) for p in parts] if cols is None else
                        [float(parts[c]) for c in cols])
    return np.asarray(vals)


def quat_xyzw_to_mat3(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R


def rot_angle(R1, R2):
    """Geodesic angle (deg) between two rotation matrices."""
    R = R1 @ R2.T
    tr = np.trace(R)
    c = max(-1.0, min(1.0, (tr - 1) / 2))
    return np.degrees(np.arccos(c))


print("=" * 70)
print("1) hand108 halves")
print("=" * 70)
lhand_gt = load_npy("lhand_raw_hand")[0]          # (108,) pre-overwrite
rhand_gt = load_npy("rhand_raw_hand")[0]          # (108,)
for side, gt, path in (("left", lhand_gt, f"{CPP}/hand108_left.txt"),
                       ("right", rhand_gt, f"{CPP}/hand108_right.txt")):
    cpp = load_txt(path).ravel()
    assert cpp.shape[0] == 108, cpp.shape
    d0 = np.abs(cpp[:54] - gt[:54]).max()
    d1 = np.abs(cpp[54:] - gt[54:]).max()
    print(f"{side}: max|diff| first half={d0:.4f}  second half={d1:.4f}")

print()
print("=" * 70)
print("2) finger-chain global rotations (C++ quats vs official joint_global_rots)")
print("=" * 70)
rgr_gt = load_npy("rhand_raw_joint_global_rots")[0]     # (127,3,3)
cpp_fq = load_txt(f"{CPP}/fingerq_right.txt")
for row in cpp_fq:
    j, q = int(row[0]), row[1:]
    R_cpp = quat_xyzw_to_mat3(q)
    ang = rot_angle(R_cpp, rgr_gt[j])
    print(f"  joint {j}: {ang:7.2f} deg")

print()
print("=" * 70)
print("3) wrist global rotation: C++ wrist_quat vs official gate_pred_global_wrist")
print("=" * 70)
pgw_gt = load_npy("gate_pred_global_wrist")[0]          # (2,3,3) [left(78), right(42)]
# C++ wristquatdbg (already mirror-corrected for left): parse from gate-debug rerun
# -- instead compare against the printed matrices captured in run output; here we
#   hardcode the values printed in the latest run (see run transcript).
cpp_wrist = {
    "left":  np.array([[0.9128, -0.3591, -0.1947],
                       [-0.3819, -0.9193, -0.0951],
                       [-0.1448, 0.1612, -0.9762]]),
    "right": np.array([[-0.2048, -0.8045, 0.5576],
                       [-0.9358, 0.3278, 0.1294],
                       [-0.2869, -0.4953, -0.8200]]),
}
for i, side in enumerate(("left", "right")):
    print(f"  {side}: {rot_angle(cpp_wrist[side], pgw_gt[i]):.2f} deg")

print()
print("=" * 70)
print("4) spliced wrist euler: C++ vs official final body_pose")
print("=" * 70)
bp_gt = load_npy("final_body_pose_params")           # (133,)
# right = [41,43,42] -> (x,z,y); left = [31,33,32] -> (x,z,y)
print(f"  official right (x,z,y): {bp_gt[41]:.4f} {bp_gt[43]:.4f} {bp_gt[42]:.4f}")
print(f"  official left  (x,z,y): {bp_gt[31]:.4f} {bp_gt[33]:.4f} {bp_gt[32]:.4f}")
print("  C++ right (wx,wz,wy):    0.2606 -0.7401  0.0832   (from splicedwristdbg)")
print("  C++ left  (wx,wz,wy):    0.1578 -0.8315 -0.7819   (from splicedwristdbg)")

print()
print("=" * 70)
print("5) scale / shape (final)")
print("=" * 70)
scale_gt = load_npy("final_scale_params")
shape_gt = load_npy("final_shape_params")
print(f"  official final scale[8]={scale_gt[8]:.4f}  scale[9]={scale_gt[9]:.4f}")
print(f"  C++      final scale8=0.0619  scale9=-0.0494  (kp2ddbg, pass-2 values)")
print(f"  official final shape[40:] = {shape_gt[40:]}")
lshape = load_npy("lhand_raw_shape")[0]
rshape = load_npy("rhand_raw_shape")[0]
print(f"  official lhand shape[40:]  = {lshape[40:]}")
print(f"  official rhand shape[40:]  = {rshape[40:]}")
lscale = load_npy("lhand_raw_scale")[0]
rscale = load_npy("rhand_raw_scale")[0]
print(f"  official lhand scale[8]={lscale[8]:.4f}  scale[9]={lscale[9]:.4f}  scale[18:]={lscale[18:]}")
print(f"  official rhand scale[8]={rscale[8]:.4f}  scale[9]={rscale[9]:.4f}  scale[18:]={rscale[18:]}")

print()
print("=" * 70)
print("6) hand crop geometry")
print("=" * 70)
bls = load_npy("batch_lhand_bbox_scale")[0, 0]
brs = load_npy("batch_rhand_bbox_scale")[0, 0]
blc = load_npy("batch_lhand_bbox_center")[0, 0]
brc = load_npy("batch_rhand_bbox_center")[0, 0]
print(f"  official lhand bbox_scale={bls} center={blc} (flipped frame)")
print(f"  official rhand bbox_scale={brs} center={brc}")
print("  C++      lhand orig_sz=145.5 orig_cx=650.5  (unflipped 1080-1-650.5=429.5)")
print("  C++      rhand orig_sz=146.0 orig_cx=384.8")
lpt = load_npy("lhand_raw_pred_cam_t")[0]
rpt = load_npy("rhand_raw_pred_cam_t")[0]
print(f"  official lhand pred_cam_t={lpt}")
print(f"  official rhand pred_cam_t={rpt}")
print("  C++      lhand pred_cam_t=(-0.0911,-0.0237,2.5162)")
print("  C++      rhand pred_cam_t=(-0.2427,-0.3224,2.2923)")
