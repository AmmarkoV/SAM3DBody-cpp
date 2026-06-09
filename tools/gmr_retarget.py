#!/usr/bin/env python3
# Retarget a LAFAN-format BVH onto a robot with GMR, using a CUSTOM ik_config
# (e.g. our position-based config) WITHOUT modifying GMR's stock files: we patch
# IK_CONFIG_DICT in memory before constructing the retargeter. Grounds the feet
# each frame (our MHR data is camera-space, not floor-aligned like real LAFAN1).
#
# Usage:
#   gmr_retarget.py --bvh F.bvh --robot unitree_g1 --config C.json \
#                   [--save out.pkl] [--video out.mp4] [--no-ground]
import argparse, pickle, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bvh", required=True)
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--config", required=True, help="custom GMR ik_config json")
ap.add_argument("--save")
ap.add_argument("--video")
ap.add_argument("--no-ground", action="store_true")
ap.add_argument("--human-height", type=float, default=0.0,
                help="override actual_human_height (m); 0 = auto-scale from skeleton")
ap.add_argument("--flip-depth", action="store_true",
                help="reverse the global front/back (depth) motion. MHR is camera-space and "
                     "its depth sign is opposite to GMR's Y-up->Z-up convention, so the robot "
                     "otherwise moonwalks. This is a RIGID per-frame translation on the depth "
                     "axis only (GMR world Y) — pose and facing are untouched.")
ap.add_argument("--no-despike", action="store_true",
                help="disable the glitch-frame filter (on by default). The filter replaces "
                     "velocity-outlier frames (tracking singularities — esp. root-rotation "
                     "flips that the CG-oriented jitter pass misses) by slerp/lerp interpolation "
                     "from clean neighbours. Important for robot actuation: a single bad frame "
                     "is a huge instantaneous joint velocity that can destabilise the robot.")
ap.add_argument("--despike-root-deg", type=float, default=40.0,
                help="root angular-velocity threshold (deg/frame) above which a frame is a glitch")
ap.add_argument("--despike-pos-m", type=float, default=0.30,
                help="per-joint position-jump threshold (m/frame) above which a frame is a glitch")
a = ap.parse_args()

from general_motion_retargeting import params
# Point the bvh_lafan1 -> <robot> config at our custom file (in-memory patch).
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = Path(a.config)

from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file

frames, h = load_bvh_file(a.bvh, format="lafan1")

# ── Glitch-frame filter (robot safety) ───────────────────────────────────────
# Monocular tracking occasionally emits a burst of garbage frames (esp. a root-
# rotation flip/singularity) that the CG-oriented offline jitter pass misses.
# For a robot, one such frame is a huge instantaneous joint velocity → it can
# fall.  We flag velocity-outlier frames and replace them with slerp/lerp
# interpolation from the nearest CLEAN frames on each side.
def _slerp_wxyz(qa, qb, t):
    a = np.array([qa[1], qa[2], qa[3], qa[0]]); b = np.array([qb[1], qb[2], qb[3], qb[0]])
    if np.dot(a, b) < 0: b = -b
    d = float(np.clip(np.dot(a, b), -1, 1))
    if d > 0.9995: r = a + t * (b - a)
    else:
        th = np.arccos(d); r = (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)
    r /= np.linalg.norm(r)
    return np.array([r[3], r[0], r[1], r[2]])

def despike_frames(frames, root_deg, pos_m):
    from scipy.spatial.transform import Rotation as R
    N = len(frames)
    if N < 3: return 0
    joints = list(frames[0].keys())
    def hq(i): a = frames[i]["Hips"][1]; return R.from_quat([a[1], a[2], a[3], a[0]])
    def jump(i, k):  # (root angVel deg, max joint pos jump m) between frames i and k
        dav = (hq(i).inv() * hq(k)).magnitude() * 180 / np.pi
        dp = max(np.linalg.norm(np.asarray(frames[k][j][0]) - np.asarray(frames[i][j][0])) for j in joints)
        return dav, dp
    v = [(0.0, 0.0)] + [jump(i - 1, i) for i in range(1, N)]   # v[i] = motion (i-1)->i
    over = lambda x: x[0] > root_deg or x[1] > pos_m
    # a frame is bad if it has a too-large jump on either side
    bad = [ (i > 0 and over(v[i])) or (i < N - 1 and over(v[i + 1])) for i in range(N) ]
    nbad = sum(bad)
    if nbad == 0: return 0
    i = 0
    while i < N:
        if not bad[i]: i += 1; continue
        a0 = i
        while i < N and bad[i]: i += 1
        b0 = i - 1                                  # bad span [a0, b0]
        lo = a0 - 1
        while lo >= 0 and bad[lo]: lo -= 1
        hi = b0 + 1
        while hi < N and bad[hi]: hi += 1
        if lo < 0 and hi >= N: break                # whole clip bad — give up
        lo_ok, hi_ok = lo >= 0, hi < N
        for t in range(a0, b0 + 1):
            if not hi_ok or not lo_ok:              # one-sided: clamp to the clean frame
                src = frames[hi if hi_ok else lo]
                for j in joints:
                    frames[t][j] = [np.asarray(src[j][0]).copy(), np.asarray(src[j][1]).copy()]
                continue
            al = (t - lo) / (hi - lo)
            for j in joints:
                p0 = np.asarray(frames[lo][j][0]); p1 = np.asarray(frames[hi][j][0])
                frames[t][j] = [(1 - al) * p0 + al * p1,
                                _slerp_wxyz(frames[lo][j][1], frames[hi][j][1], al)]
    return nbad

if not a.no_despike:
    n = despike_frames(frames, a.despike_root_deg, a.despike_pos_m)
    if n: print(f"[despike] replaced {n} glitch frame(s) of {len(frames)} by interpolation")

# ── Scale compensation ────────────────────────────────────────────────────────
# GMR scales targets by (actual_human_height / 1.8) * human_scale_table, and that
# table is calibrated for a real-LAFAN1-size skeleton (bone-path ~1.665 m, height
# 1.75 m). Our MHR skeleton is usually smaller, so the loader's hard-coded 1.75 m
# under-scales it and the robot crouches. Measure our skeleton's intrinsic
# (pose-independent) bone-path height and pass an actual_human_height that makes
# GMR scale it like a real LAFAN1 skeleton: actual = 1.75 * 1.665 / H_ours.
REF_BONEPATH_M, REF_HEIGHT_M = 1.665, 1.75
def _bone_path_height(fr):
    def d(j1, j2): return float(np.linalg.norm(np.asarray(fr[j1][0]) - np.asarray(fr[j2][0])))
    spine = d("Hips","Spine") + d("Spine","Spine1") + d("Spine1","Spine2") + d("Spine2","Neck") + d("Neck","Head")
    leg   = d("Hips","LeftUpLeg") + d("LeftUpLeg","LeftLeg") + d("LeftLeg","LeftFoot")
    return spine + leg
if a.human_height > 0:
    human_height = a.human_height
else:
    H = _bone_path_height(frames[0])
    human_height = REF_HEIGHT_M * REF_BONEPATH_M / max(H, 1e-6)
    print(f"[scale] skeleton bone-path={H:.3f} m -> actual_human_height={human_height:.3f} m")

rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=human_height)
viewer = RobotMotionViewer(robot_type=a.robot, motion_fps=30,
                           record_video=bool(a.video), video_path=a.video or "")

# ── depth (front/back) correction ───────────────────────────────────────────
# Reverse the global depth motion by rigidly translating every joint on GMR's
# depth axis (world Y) about frame 0, so frame 0 stays put and the drift flips.
# Rigid → pose, facing and the saved dof are all unchanged; only where the body
# sits in depth changes. Keeps the human overlay consistent (applied pre-retarget).
DEPTH = 1  # GMR world Y after the Y-up->Z-up load (X=L/R, Y=depth, Z=up)
_y0 = float(np.asarray(frames[0]["Hips"][0])[DEPTH]) if a.flip_depth else 0.0
def _flip_depth(fr):
    shift = -2.0 * (float(np.asarray(fr["Hips"][0])[DEPTH]) - _y0)
    out = {}
    for j, (p, q) in fr.items():
        p = np.asarray(p, dtype=float).copy(); p[DEPTH] += shift
        out[j] = [p, q]
    return out

qpos_list = []
for fr in frames:
    if a.flip_depth:
        fr = _flip_depth(fr)
    q = rt.retarget(fr, offset_to_ground=not a.no_ground)
    qpos_list.append(q.copy())
    viewer.step(root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:],
                human_motion_data=rt.scaled_human_data, rate_limit=False, follow_camera=True)
viewer.close()

if a.save:
    qp = np.array(qpos_list)
    motion = {"fps": 30,
              "root_pos": qp[:, :3],
              "root_rot": qp[:, 3:7][:, [1, 2, 3, 0]],  # wxyz -> xyzw
              "dof_pos": qp[:, 7:],
              "local_body_pos": None, "link_body_list": None}
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(motion, open(a.save, "wb"))
    print("saved", a.save)
print("done")
