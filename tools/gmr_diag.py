#!/usr/bin/env python3
"""
gmr_diag.py — quantitative diagnostics for a GMR retarget (no rendering).

Headlessly retargets a lafan_mhr BVH and reports, per matched joint, how far the
robot link lands from its (scaled) human target, plus a few pose sanity checks:
  • per-joint position residual (robot link origin vs human target)
  • root pelvis height (mean/min/max) — sanity vs the robot's nominal stance
  • elbow flexion: human target vs robot achieved (catches "arms too long/short")
  • foot/leg cross check — left ankle should sit on the robot's left

Use it to compare config tweaks without watching video.  Note: a residual is
robot-vs-target, so the large torso/(dropped) proximal-link numbers reflect the
robot link origin NOT coinciding with the human joint — that's expected, not a
regression (see GMR.md).  Watch the DELTAS between configs, and the elbow angle.

Run from the GMR dir with its venv:
    GMR/venv/bin/python tools/gmr_diag.py --bvh gmr_out/<clip>/<clip>_0.bvh
    GMR/venv/bin/python tools/gmr_diag.py --bvh ... --config /tmp/experiment.json
"""
import argparse
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bvh", required=True)
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--config", default=None, help="defaults to the repo position config")
ap.add_argument("--human-height", type=float, default=0.0)
a = ap.parse_args()

HERE = Path(__file__).resolve().parent
CFG = Path(a.config) if a.config else HERE.parent / "scripts" / "gmr_configs" / "bvh_lafan1pos_to_g1.json"

from general_motion_retargeting import params
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = CFG
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file

frames, _ = load_bvh_file(a.bvh, format="lafan1")
def _bone_path(fr):
    d = lambda j1, j2: float(np.linalg.norm(np.asarray(fr[j1][0]) - np.asarray(fr[j2][0])))
    return (d("Hips","Spine")+d("Spine","Spine1")+d("Spine1","Spine2")+d("Spine2","Neck")+d("Neck","Head")
            +d("Hips","LeftUpLeg")+d("LeftUpLeg","LeftLeg")+d("LeftLeg","LeftFoot"))
ahh = a.human_height if a.human_height > 0 else 1.75 * 1.665 / max(_bone_path(frames[0]), 1e-6)
rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=ahh, verbose=False)

tbl = rt.ik_match_table2
frame_for = {e[0]: f for f, e in tbl.items() if e[1] or e[2]}
IDX = range(0, len(frames), max(1, len(frames)//60))

def angle(a_, b_, c_):
    u = a_-b_; v = c_-b_
    return np.degrees(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9), -1, 1)))
def rp(n): return rt.configuration.data.xpos[rt.robot_body_names[n]]

res = {b: [] for b in frame_for}
zs, h_elb, r_elb, cross = [], [], [], []
for i in IDX:
    q = rt.retarget(frames[i], offset_to_ground=True); sd = rt.scaled_human_data
    zs.append(q[2])
    for b, f in frame_for.items():
        res[b].append(np.linalg.norm(rt.configuration.data.xpos[rt.robot_body_names[f]] - np.asarray(sd[b][0])))
    fr = frames[i]
    h_elb.append(angle(np.asarray(fr["LeftArm"][0]), np.asarray(fr["LeftForeArm"][0]), np.asarray(fr["LeftHand"][0])))
    r_elb.append(angle(rp("left_shoulder_yaw_link"), rp("left_elbow_link"), rp("left_wrist_yaw_link")))
    xm = rt.configuration.data.xmat[rt.robot_body_names["pelvis"]].reshape(3,3)
    cross.append(np.dot(rp("left_ankle_roll_link")-rp("right_ankle_roll_link"), xm@[0,1,0.]))

print(f"config = {CFG.name}   actual_human_height = {ahh:.3f}\n")
print(f"{'human_body':14s} {'robot_frame':24s} {'mean':>7s} {'max':>7s}")
tot = []
for b in frame_for:
    arr = np.array(res[b]); tot += list(arr)
    print(f"{b:14s} {frame_for[b]:24s} {arr.mean():7.4f} {arr.max():7.4f}")
print(f"\nOVERALL residual mean = {np.mean(tot):.4f} m")
print(f"root pelvis Z: mean={np.mean(zs):.3f} min={np.min(zs):.3f} max={np.max(zs):.3f} m")
print(f"left elbow flexion (180=straight): human={np.mean(h_elb):.0f}deg  robot={np.mean(r_elb):.0f}deg")
print(f"left-vs-right ankle separation along pelvis-left: {np.mean(cross):+.3f} m  (>0 = uncrossed)")
