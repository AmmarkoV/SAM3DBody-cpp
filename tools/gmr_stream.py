#!/usr/bin/env python3
# Live webcam -> humanoid robot, streaming counterpart of tools/gmr_retarget.py.
#
# fast_sam_3dbody_run --bvh-stream emits ONE BVH MOTION line per frame (prefixed
# "@F ") for a single tracked person.  This driver reads those lines on stdin,
# reassembles a 1-frame BVH by prepending the lafan_mhr.bvh HIERARCHY header, and
# runs GMR's OWN load_bvh_file on it -- so BOTH the C++ bvh_writer rotation math
# and GMR's exact LAFAN forward-kinematics / Y-up->Z-up / cm->m transform are
# reused verbatim (no reimplementation).  Each frame is then retargeted, causally
# despiked (robot safety), and pushed to a pluggable Sink.
#
# Unlike gmr_retarget.py this is CAUSAL: single pass, no look-ahead.  The offline
# despike_qpos interpolates a glitch frame from its FUTURE clean neighbour, which
# we cannot do live; instead we CLAMP each frame's step away from the last
# accepted qpos so instantaneous joint velocity stays bounded (the same safety
# goal, one-frame history).
#
# Usage (normally via scripts/webcam_gmr.sh):
#   fast_sam_3dbody_run ... --bvh-template lafan_mhr.bvh --bvh-stream - \
#     | gmr_stream.py --robot unitree_g1 --config <pos_config.json> --flip-depth
import argparse, copy, sys, tempfile
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--config", required=True, help="custom GMR ik_config json (position-based)")
ap.add_argument("--template", default=str(REPO / "lafan_mhr.bvh"),
                help="BVH template whose HIERARCHY header the streamed lines are appended to")
ap.add_argument("--sink", choices=["viewer", "dds"], default="viewer",
                help="viewer = live MuJoCo RobotMotionViewer; dds = Unitree DDS (unitree_mujoco / real G1) [stub]")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--flip-depth", action="store_true",
                help="reverse global front/back drift (MHR camera-space depth sign is opposite to "
                     "GMR's Y-up->Z-up convention). Rigid per-frame depth-axis shift about frame 0.")
ap.add_argument("--no-ground", action="store_true", help="disable per-frame foot grounding")
ap.add_argument("--human-height", type=float, default=0.0,
                help="override actual_human_height (m); 0 = auto-scale from the first frame's skeleton")
# Causal despike (robot safety) -- same thresholds as gmr_retarget.py, clamped not interpolated.
ap.add_argument("--no-despike", action="store_true",
                help="disable the causal step clamp (on by default). The clamp bounds per-frame "
                     "root/joint velocity so a tracking singularity cannot command a huge instantaneous "
                     "motion that destabilises the robot.")
ap.add_argument("--despike-root-deg", type=float, default=40.0,
                help="max root angular velocity (deg/frame) before the step is clamped")
ap.add_argument("--despike-pos-m", type=float, default=0.30,
                help="max root position velocity (m/frame) before the step is clamped")
ap.add_argument("--despike-dof-deg", type=float, default=45.0,
                help="max per-dof angular velocity (deg/frame) before the step is clamped")
a = ap.parse_args()

# Point the bvh_lafan1 -> <robot> config at our custom file (in-memory patch;
# GMR's stock files are never touched).  Same trick as gmr_retarget.py.
from general_motion_retargeting import params
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = Path(a.config)

from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scipy.spatial.transform import Rotation as R


# ── helpers lifted from tools/gmr_retarget.py (kept identical) ───────────────
def _slerp_wxyz(qa, qb, t):
    a_ = np.array([qa[1], qa[2], qa[3], qa[0]]); b_ = np.array([qb[1], qb[2], qb[3], qb[0]])
    if np.dot(a_, b_) < 0: b_ = -b_
    d = float(np.clip(np.dot(a_, b_), -1, 1))
    if d > 0.9995: r = a_ + t * (b_ - a_)
    else:
        th = np.arccos(d); r = (np.sin((1 - t) * th) * a_ + np.sin(t * th) * b_) / np.sin(th)
    r /= np.linalg.norm(r)
    return np.array([r[3], r[0], r[1], r[2]])

REF_BONEPATH_M, REF_HEIGHT_M = 1.665, 1.75
def _bone_path_height(fr):
    def d(j1, j2): return float(np.linalg.norm(np.asarray(fr[j1][0]) - np.asarray(fr[j2][0])))
    spine = d("Hips","Spine") + d("Spine","Spine1") + d("Spine1","Spine2") + d("Spine2","Neck") + d("Neck","Head")
    leg   = d("Hips","LeftUpLeg") + d("LeftUpLeg","LeftLeg") + d("LeftLeg","LeftFoot")
    return spine + leg

DEPTH = 1  # GMR world Y after the Y-up->Z-up load (X=L/R, Y=depth, Z=up)
def _flip_depth(fr, y0):
    shift = -2.0 * (float(np.asarray(fr["Hips"][0])[DEPTH]) - y0)
    out = {}
    for j, (p, q) in fr.items():
        p = np.asarray(p, dtype=float).copy(); p[DEPTH] += shift
        out[j] = [p, q]
    return out


# ── causal despike: bound the per-frame step away from the last accepted qpos ─
# The offline despike replaces a glitch frame by interpolating its FUTURE clean
# neighbour; live we have no future, so we instead scale the step from prev->q so
# no channel's velocity exceeds its cap.  A single tracking singularity is thus
# spread over several frames instead of commanded as one huge jump.
def causal_clamp(prev, q, root_deg, pos_m, dof_deg):
    rp = R.from_quat([prev[4], prev[5], prev[6], prev[3]])
    rq = R.from_quat([q[4], q[5], q[6], q[3]])
    dav = (rp.inv() * rq).magnitude() * 180 / np.pi
    dp  = float(np.linalg.norm(q[:3] - prev[:3]))
    dd  = float(np.max(np.abs(q[7:] - prev[7:]))) * 180 / np.pi
    t = 1.0
    if dav > root_deg: t = min(t, root_deg / dav)
    if dp  > pos_m:    t = min(t, pos_m / dp)
    if dd  > dof_deg:  t = min(t, dof_deg / dd)
    if t >= 1.0:
        return q, False
    out = q.copy()
    out[:3]  = prev[:3] + t * (q[:3] - prev[:3])
    out[7:]  = prev[7:] + t * (q[7:] - prev[7:])
    out[3:7] = _slerp_wxyz(prev[3:7], q[3:7], t)
    return out, True


# ── pluggable sinks ──────────────────────────────────────────────────────────
class ViewerSink:
    """Live MuJoCo visualisation of the retargeted robot (GMR's own viewer)."""
    def __init__(self, robot, fps):
        self.v = RobotMotionViewer(robot_type=robot, motion_fps=fps)
    def step(self, q, overlay):
        self.v.step(root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:],
                    human_motion_data=overlay, rate_limit=True, follow_camera=True)
    def close(self):
        self.v.close()

class DDSSink:
    """Publish the qpos reference (root_pos/root_rot/dof_pos) over Unitree's DDS
    interface via unitree_sdk2_python -- the SAME interface unitree_mujoco (a
    physics sim of the G1) and the real robot both expose, flipped by a SIM flag
    (cf. Robotics-Ark/ark_unitree_g1). No custom low-level protocol is written.

    NOT IMPLEMENTED HERE. Two things must be added before this drives hardware:
      1. A balance-aware whole-body TRACKING POLICY. GMR qpos is a *kinematic
         reference*, not directly executable -- open-loop playback topples a
         free-standing G1. A policy (e.g. BeyondMimic trained on the retargeted
         motion) turns this reference into stable motor commands. GMR is designed
         to FEED such a policy, not replace it.
      2. Joint position/velocity/torque limit enforcement + an e-stop path, and
         hold/interpolation up to the controller rate (this stream is ~4-6 fps,
         backbone-bound; a robot controller wants a steady higher rate).
    The `step(qpos)` seam below is where a `unitree_sdk2_python` publisher goes."""
    def __init__(self, robot, fps):
        raise NotImplementedError(
            "DDSSink is a stub. It publishes over Unitree's DDS interface "
            "(unitree_sdk2_python / unitree_mujoco), but needs a whole-body tracking "
            "policy + safety layer first -- see the class docstring and GMR.md.")
    def step(self, q, overlay): ...
    def close(self): ...


# ── template header: everything up to & including the 'Frame Time:' line ─────
header = []
with open(a.template) as f:
    for line in f:
        s = line.rstrip("\n")
        header.append("Frames: 1" if s.startswith("Frames:") else s)
        if s.startswith("Frame Time:"):
            break
HEADER = "\n".join(header) + "\n"

_tmp = tempfile.NamedTemporaryFile("w", suffix=".bvh", delete=False)
TMP_PATH = _tmp.name
_tmp.close()

def frame_from_line(motion_line):
    with open(TMP_PATH, "w") as fh:
        fh.write(HEADER)
        fh.write(motion_line)
        fh.write("\n")
    frames, _ = load_bvh_file(TMP_PATH, format="lafan1")
    return frames[0]


# ── main streaming loop ──────────────────────────────────────────────────────
def make_sink():
    return DDSSink(a.robot, a.fps) if a.sink == "dds" else ViewerSink(a.robot, a.fps)

def main():
    rt = None
    sink = None
    prev_q = None
    y0 = 0.0
    nclamp = 0
    try:
        for raw in sys.stdin:
            raw = raw.rstrip("\n")
            if not raw.startswith("@F "):
                if raw:                       # forward the binary's diagnostics
                    print(raw, file=sys.stderr)
                continue

            fr = frame_from_line(raw[3:])

            if rt is None:                    # lazy init on the first real frame
                if a.human_height > 0:
                    hh = a.human_height
                else:
                    H = _bone_path_height(fr)
                    hh = REF_HEIGHT_M * REF_BONEPATH_M / max(H, 1e-6)
                    print(f"[gmr_stream] bone-path={H:.3f} m -> actual_human_height={hh:.3f} m",
                          file=sys.stderr)
                rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=hh)
                sink = make_sink()
                if a.flip_depth:
                    y0 = float(np.asarray(fr["Hips"][0])[DEPTH])

            if a.flip_depth:
                fr = _flip_depth(fr, y0)

            q = rt.retarget(fr, offset_to_ground=not a.no_ground)

            if not a.no_despike and prev_q is not None:
                q, clamped = causal_clamp(prev_q, q, a.despike_root_deg,
                                          a.despike_pos_m, a.despike_dof_deg)
                nclamp += int(clamped)
            prev_q = q.copy()

            sink.step(q, copy.deepcopy(rt.scaled_human_data))
    except KeyboardInterrupt:
        pass
    finally:
        if sink is not None:
            sink.close()
        if nclamp:
            print(f"[gmr_stream] causal despike clamped {nclamp} frame(s)", file=sys.stderr)

if __name__ == "__main__":
    main()
