#!/usr/bin/env python3
"""
gmr_solve_offsets.py — re-solve the GMR position-config rotation offsets from a clip.

The position-based GMR config (scripts/gmr_configs/bvh_lafan1pos_to_g1.json) keeps
a handful of *orientation* offsets that position alone can't pin down:
  • pelvis  — body facing
  • feet    — foot yaw (kept flat)
  • wrists  — hand twist (the elbow-flip disambiguator)
These were solved from a calibration clip; this tool re-derives them from any
lafan_mhr BVH so you can retune (e.g. for a new robot, or if a clip exposes a
better calibration).  It prints values ready to paste into the config.

Method (all empirical, no hand-tuned quaternions):
  • pelvis : sweep the 24 axis-aligned offsets, pick the one that is upright,
             leaves the legs UN-crossed, and has the lowest FK residual.
  • feet   : solve the constant offset mapping the human foot quat -> a FLAT foot
             frame pointing along the (horizontal) toe direction.
  • arms   : solve the constant offset mapping each arm bone quat -> a frame built
             from limb geometry (segment dir + elbow-plane normal; finger+thumb for
             the hand).  Right side uses negated plane-normal / thumb so the
             handedness matches the left (the cross products flip L<->R otherwise).

Run from the GMR dir with its venv:
    GMR/venv/bin/python tools/gmr_solve_offsets.py --bvh gmr_out/<clip>/<clip>_0.bvh
"""
import argparse, itertools, json, copy
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("--bvh", required=True, help="a lafan_mhr-template BVH (calibration clip)")
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--config", default=None, help="base config (defaults to the repo position config)")
ap.add_argument("--human-height", type=float, default=0.0, help="override actual_human_height (0=auto)")
a = ap.parse_args()

HERE = Path(__file__).resolve().parent
DEF_CFG = HERE.parent / "scripts" / "gmr_configs" / "bvh_lafan1pos_to_g1.json"
CFG = Path(a.config) if a.config else DEF_CFG

from general_motion_retargeting import params
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = CFG
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file

frames, _ = load_bvh_file(a.bvh, format="lafan1")

# auto human height (same rule as tools/gmr_retarget.py)
def _bone_path(fr):
    d = lambda j1, j2: float(np.linalg.norm(np.asarray(fr[j1][0]) - np.asarray(fr[j2][0])))
    return (d("Hips","Spine")+d("Spine","Spine1")+d("Spine1","Spine2")+d("Spine2","Neck")+d("Neck","Head")
            +d("Hips","LeftUpLeg")+d("LeftUpLeg","LeftLeg")+d("LeftLeg","LeftFoot"))
ahh = a.human_height if a.human_height > 0 else 1.75 * 1.665 / max(_bone_path(frames[0]), 1e-6)
IDX = range(0, len(frames), max(1, len(frames)//25))

def wxyz(q):  # scipy xyzw -> json wxyz, rounded
    return [round(float(q[3]),4), round(float(q[0]),4), round(float(q[1]),4), round(float(q[2]),4)]

# ── 1) pelvis: sweep the 24 proper axis-aligned offsets ──────────────────────
def proper_rots():
    seen, out = {}, []
    for e in itertools.product([0,90,180,270], repeat=3):
        r = R.from_euler('XYZ', e, degrees=True); m = np.round(r.as_matrix()).astype(int)
        k = m.tobytes()
        if k not in seen:
            seen[k] = 1; out.append(r.as_quat())
    return out

def solve_pelvis():
    base = json.load(open(CFG)); rows = []
    for q in proper_rots():
        cfg = copy.deepcopy(base)
        for t in ("ik_match_table1","ik_match_table2"):
            cfg[t]["pelvis"][4] = wxyz(q)
        p = Path("/tmp/_solve_pelvis.json"); json.dump(cfg, open(p,"w"))
        params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = p
        rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=ahh, verbose=False)
        bP, bL, bR = (rt.robot_body_names[n] for n in ("pelvis","left_ankle_roll_link","right_ankle_roll_link"))
        tbl = rt.ik_match_table2; ff = {e[0]: f for f, e in tbl.items() if e[1] or e[2]}
        cross, up, resid = [], [], []
        for i in IDX:
            rt.retarget(frames[i], offset_to_ground=True); sd = rt.scaled_human_data
            xm = rt.configuration.data.xmat[bP].reshape(3,3)
            cross.append(np.dot(rt.configuration.data.xpos[bL]-rt.configuration.data.xpos[bR], xm@[0,1,0.]))
            up.append((xm@[0,0,1.])[2])
            for b, f in ff.items():
                resid.append(np.linalg.norm(rt.configuration.data.xpos[rt.robot_body_names[f]]-np.asarray(sd[b][0])))
        rows.append((np.mean(resid), np.mean(cross), np.mean(up), wxyz(q)))
    rows = [r for r in rows if r[1] > 0.02 and r[2] > 0.8]  # uncrossed + upright
    rows.sort()
    params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = CFG
    return rows[0][3] if rows else None

# ── 2) feet: constant offset to a flat foot pointing along the toe direction ──
def avg_offset(get_human_R, get_desired_R):
    offs = []
    for fr in frames:
        D = get_desired_R(fr)
        if D is None: continue
        H = get_human_R(fr)
        offs.append(R.from_matrix(H.T @ D).as_quat())
    offs = np.array(offs); offs[(offs @ offs[0]) < 0] *= -1
    m = offs.mean(0); m /= np.linalg.norm(m)
    dev = np.array([(R.from_quat(o*np.sign(o@m)).inv()*R.from_quat(m)).magnitude() for o in offs])*180/np.pi
    return wxyz(m), dev.mean()

def hquat(fr, j):
    q = fr[j][1]; return R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def P(fr, j): return np.asarray(fr[j][0])
def frame_xz(x, zish):
    x = x/(np.linalg.norm(x)+1e-9); y = np.cross(zish, x); n = np.linalg.norm(y)
    if n < 1e-4: return None
    y /= n; return np.column_stack([x, y, np.cross(x, y)])

def solve_foot(side):
    def D(fr):
        f = P(fr, f"{side}Toe") - P(fr, f"{side}Foot"); f[2] = 0.0  # flat
        if np.linalg.norm(f) < 1e-3: return None
        return frame_xz(f, [0,0,1.])
    return avg_offset(lambda fr: hquat(fr, f"{side}FootMod"), D)

# ── 3) arms: segment dir + elbow-plane normal (hand: finger + thumb) ──────────
def solve_arm(side):
    s, e, w = f"{side}Arm", f"{side}ForeArm", f"{side}Hand"
    mid, th = f"{side}HandMiddle1", f"{side}HandThumb1"
    sgn = 1.0 if side == "Left" else -1.0   # flip handedness for the right side
    def planeN(fr):
        n = sgn*np.cross(P(fr,e)-P(fr,s), P(fr,w)-P(fr,e)); return n/(np.linalg.norm(n)+1e-9)
    o_s = avg_offset(lambda fr: hquat(fr,s), lambda fr: frame_xz(P(fr,e)-P(fr,s), planeN(fr)))
    o_e = avg_offset(lambda fr: hquat(fr,e), lambda fr: frame_xz(P(fr,w)-P(fr,e), planeN(fr)))
    o_w = avg_offset(lambda fr: hquat(fr,w), lambda fr: frame_xz(P(fr,mid)-P(fr,w), sgn*(P(fr,th)-P(fr,w))))
    return o_s, o_e, o_w

print(f"[solve] bvh={a.bvh}  robot={a.robot}  actual_human_height={ahh:.3f}\n")
pel = solve_pelvis()
print(f"pelvis rot_offset (wxyz): {pel}\n")
for side in ("Left","Right"):
    q, dev = solve_foot(side); print(f"{side} ankle rot_offset: {q}   (dev {dev:.1f}deg)")
print()
for side in ("Left","Right"):
    (qs,ds),(qe,de),(qw,dw) = solve_arm(side)
    print(f"{side} shoulder: {qs} (dev {ds:.1f})   elbow: {qe} (dev {de:.1f})   wrist: {qw} (dev {dw:.1f})")
print("\nPaste the pelvis/ankle/wrist offsets into both ik_match_table1 and ik_match_table2.")
print("(shoulder/elbow offsets are printed for reference; the shipped config keeps")
print(" shoulder dropped and elbow position-only — enable them only if you add weight.)")
