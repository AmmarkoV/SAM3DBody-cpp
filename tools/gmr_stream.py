#!/usr/bin/env python3
# Live webcam -> humanoid robot, streaming counterpart of tools/gmr_retarget.py.
#
# fast_sam_3dbody_run --bvh-stream emits ONE BVH MOTION line per frame (prefixed
# "@F ") for a single tracked person.  This driver reads those lines on stdin,
# reassembles a 1-frame BVH by prepending the bvh/lafan_mhr.bvh HIERARCHY header, and
# runs GMR's OWN load_bvh_file on it -- so BOTH the C++ bvh_writer rotation math
# and GMR's exact LAFAN forward-kinematics / Y-up->Z-up / cm->m transform are
# reused verbatim (no reimplementation).  Each frame is then retargeted, causally
# despiked (robot safety), and pushed to a pluggable Sink.
#
# GMR's IK is warm-started (retarget() integrates into rt.configuration in place),
# which a crowd can walk into a bent local minimum it never climbs out of, so two
# escapes re-home it to the robot's neutral qpos0: pressing R in the robot window,
# and an automatic residual check.  A third case is losing the person entirely --
# the C++ side then streams nothing at all -- so an idle watchdog eases the robot
# home rather than leaving it frozen.  See "Stuck-pose recovery" in GMR.md.
#
# Unlike gmr_retarget.py this is CAUSAL: single pass, no look-ahead.  The offline
# despike_qpos interpolates a glitch frame from its FUTURE clean neighbour, which
# we cannot do live; instead we CLAMP each frame's step away from the last
# accepted qpos so instantaneous joint velocity stays bounded (the same safety
# goal, one-frame history).
#
# Usage (normally via scripts/webcam_gmr.sh):
#   fast_sam_3dbody_run ... --bvh-template bvh/lafan_mhr.bvh --bvh-stream - \
#     | gmr_stream.py --robot unitree_g1 --config <pos_config.json> --flip-depth
import argparse, copy, os, select, sys, time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--config", required=True, help="custom GMR ik_config json (position-based)")
ap.add_argument("--template", default=str(REPO / "bvh/lafan_mhr.bvh"),
                help="BVH template whose HIERARCHY header the streamed lines are appended to")
ap.add_argument("--sink", choices=["viewer", "dds"], default="viewer",
                help="viewer = live MuJoCo RobotMotionViewer; dds = Unitree DDS (unitree_mujoco / real G1) [stub]")
ap.add_argument("--fps", type=int, default=30)
# Live shared-memory transport (SharedMemoryVideoBuffers).  When --bvh-shm names
# a POSIX shm descriptor, frames are read from it instead of "@F" lines on stdin
# (no ASCII pipe).  Empty = the stdin fallback.  Either way the disk-free FK below
# is used, so no per-frame temp .bvh is ever written.
ap.add_argument("--bvh-shm", default="",
                help="POSIX shm descriptor to read frames from (SharedMemoryVideoBuffers); "
                     "empty = read @F lines from stdin")
ap.add_argument("--shm-stream", default="bvh", help="feed name within --bvh-shm")
ap.add_argument("--shm-lib", default="", help="path to libSharedMemoryVideoBuffers.so "
                "(default: <repo>/src/SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so)")
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
# Stuck-pose recovery.  GMR's IK is a warm-started differential solver: retarget()
# integrates velocities into rt.configuration IN PLACE, so every frame starts from
# wherever the last one ended.  With a crowd, --max-persons 1 can hand it a
# different person frame to frame; the contradictory targets can walk it into a
# bent local minimum it cannot climb out of, and it then stays bent even after the
# source pose is clean again.  Both escapes below re-home the solver to the robot
# model's neutral qpos0 and re-solve the SAME targets from there.
ap.add_argument("--no-reset-on-stuck", action="store_true",
                help="disable the automatic escape (on by default). When on, a persistently high "
                     "IK residual triggers a re-solve from the neutral pose, kept only if it is "
                     "actually better than the warm-started one (so a false trigger costs one "
                     "extra IK solve and changes nothing).")
ap.add_argument("--reset-stuck-err", type=float, default=4.0,
                help="IK residual (sum of the two match tables' task-error norms) above which a "
                     "frame counts as stuck. Measured with the shipped g1 config: healthy tracking "
                     "sits at 1.6-2.0 and peaks at 3.0, while real stuck episodes read 8-19.")
ap.add_argument("--reset-stuck-frames", type=int, default=2,
                help="consecutive over-threshold frames before the escape is attempted")
ap.add_argument("--reset-idle-s", type=float, default=2.0,
                help="seconds without a frame after which the robot is eased back to the neutral "
                     "pose (0 disables). The C++ side streams nothing while nobody is detected, so "
                     "a crowd that loses the tracked person would otherwise leave the robot frozen "
                     "in whatever pose the last frame happened to be.")
a = ap.parse_args()

# Point the bvh_lafan1 -> <robot> config at our custom file (in-memory patch;
# GMR's stock files are never touched).  Same trick as gmr_retarget.py.
from general_motion_retargeting import params
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = Path(a.config)

from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
from scipy.spatial.transform import Rotation as R
from lafan_fk import LafanFK   # disk-free equivalent of load_bvh_file (tools/)


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


# ── stuck-pose recovery ──────────────────────────────────────────────────────
# Set from the MuJoCo viewer's UI thread, consumed by the streaming loop (a plain
# bool is enough: both sides are Python bytecode under the GIL).
_reset_requested = False

RESET_KEY = "R"
def _on_key(keycode):
    """RobotMotionViewer -> mujoco.viewer key_callback. GLFW reports letter keys
    as their uppercase ASCII code; accept both spellings anyway."""
    global _reset_requested
    if keycode in (ord(RESET_KEY), ord(RESET_KEY.lower())):
        _reset_requested = True


# How often the frame sources wake up to report "still nothing" (seconds).  Small
# enough that the idle ease looks smooth, large enough not to spin.
_POLL_S = 0.02


def ease_step(prev, target):
    """One velocity-bounded step from prev toward target, reusing the causal
    despike caps -- so easing home is as safe to actuate as tracked motion, and
    repeated calls land exactly on target."""
    if a.no_despike:
        return target.copy()
    return causal_clamp(prev, target, a.despike_root_deg, a.despike_pos_m, a.despike_dof_deg)[0]


def ik_residual(rt):
    """Total IK task error of the pose currently in rt.configuration. Same
    quantity GMR reports internally, summed over whichever match tables are on."""
    e = 0.0
    if rt.use_ik_match_table1: e += float(rt.error1())
    if rt.use_ik_match_table2: e += float(rt.error2())
    return e


# ── pluggable sinks ──────────────────────────────────────────────────────────
class ViewerSink:
    """Live MuJoCo visualisation of the retargeted robot (GMR's own viewer)."""
    def __init__(self, robot, fps, key_callback=None):
        self.v = RobotMotionViewer(robot_type=robot, motion_fps=fps,
                                   keyboard_callback=key_callback)
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


# ── disk-free frame decode ───────────────────────────────────────────────────
# Parse the template HIERARCHY ONCE, then turn each frame's channel floats
# straight into GMR's {joint:(pos,quat)} dict via GMR's own quat_fk.  This is
# byte-for-byte what load_bvh_file(format="lafan1") returned, but with no temp
# .bvh and no per-frame HIERARCHY re-parse (tools/lafan_fk.py).
fk = LafanFK(a.template, fmt="lafan1")

def channels_from_line(motion_line):
    return np.array(motion_line.split(), dtype=np.float64)


# ── main streaming loop ──────────────────────────────────────────────────────
def make_sink():
    if a.sink == "dds":
        return DDSSink(a.robot, a.fps)          # no viewer, hence no reset key
    return ViewerSink(a.robot, a.fps, _on_key)

# ── frame sources: shared memory (fast path) or @F lines on stdin (fallback) ──
def _stdin_channels():
    """Yield channel arrays from '@F' lines on stdin, or None when _POLL_S passes
    with nothing to read (which is what drives the idle watchdog).

    We read the fd ourselves and split lines by hand rather than iterating
    sys.stdin: select() reports the OS-level fd, so a TextIOWrapper sitting on a
    chunk of already-buffered lines would look idle when it is not.
    """
    fd  = sys.stdin.fileno()
    buf = b""
    while True:
        if b"\n" not in buf:
            if not select.select([fd], [], [], _POLL_S)[0]:
                yield None
                continue
            chunk = os.read(fd, 65536)
            if not chunk:                     # EOF -- the producer is gone
                return
            buf += chunk
            continue
        line, buf = buf.split(b"\n", 1)
        raw = line.decode(errors="replace").rstrip("\r")
        if not raw.startswith("@F "):
            if raw:                           # forward the binary's diagnostics
                print(raw, file=sys.stderr)
            continue
        yield channels_from_line(raw[3:])


def main():
    global _reset_requested
    rt = None
    sink = None
    prev_q = None
    y0 = 0.0
    nclamp = 0
    reader = None
    q_home = None                 # the robot's neutral qpos0 -- what we reset to
    stuck_n = 0                   # consecutive over-threshold frames
    nrehome = 0
    t_last_frame = time.monotonic()
    easing = False                # currently walking the robot back to neutral
    nidle = 0

    if a.bvh_shm:
        from shm_bvh_reader import ShmBvhReader
        lib = a.shm_lib or str(REPO / "src" / "SharedMemoryVideoBuffers" / "libSharedMemoryVideoBuffers.so")
        reader = ShmBvhReader(lib, a.bvh_shm, a.shm_stream)
        print(f"[gmr_stream] reading frames from shm '{a.bvh_shm}:{a.shm_stream}'", file=sys.stderr)
        def source():
            while True:                       # None on timeout = an idle tick
                yield reader.read(timeout=_POLL_S)
        channels = source()
    else:
        channels = _stdin_channels()

    try:
        for arr in channels:
            if arr is None:                   # no frame this tick -- idle watchdog
                if rt is None:                # nothing has ever arrived; no viewer yet
                    continue
                idle_s = time.monotonic() - t_last_frame
                if not (_reset_requested or
                        (a.reset_idle_s > 0 and idle_s > a.reset_idle_s)):
                    continue
                if not easing:
                    easing = True
                    why = "reset key" if _reset_requested else f"no frames for {idle_s:.1f}s"
                    _reset_requested = False
                    # Re-home the solver too, so tracking resumes from neutral
                    # rather than snapping back to the pose we just walked away from.
                    rt.configuration.update(q_home)
                    nidle += 1
                    print(f"[gmr_stream] {why} -> easing to the neutral pose", file=sys.stderr)
                prev_q = q_home.copy() if prev_q is None else ease_step(prev_q, q_home)
                sink.step(prev_q, {})         # {} (not None) clears the stale human overlay
                continue

            t_last_frame = time.monotonic()
            easing = False
            fr = fk.frame_from_channels(arr)

            if rt is None:                    # lazy init on the first real frame
                if a.human_height > 0:
                    hh = a.human_height
                else:
                    H = _bone_path_height(fr)
                    hh = REF_HEIGHT_M * REF_BONEPATH_M / max(H, 1e-6)
                    print(f"[gmr_stream] bone-path={H:.3f} m -> actual_human_height={hh:.3f} m",
                          file=sys.stderr)
                rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=hh)
                # mink.Configuration is built from the model's default qpos0, and
                # nothing has integrated into it yet, so this IS the neutral pose.
                q_home = rt.configuration.q.copy()
                sink = make_sink()
                if a.sink != "dds":
                    print(f"[gmr_stream] press {RESET_KEY} in the robot window to reset the pose",
                          file=sys.stderr)
                if a.flip_depth:
                    y0 = float(np.asarray(fr["Hips"][0])[DEPTH])

            if a.flip_depth:
                fr = _flip_depth(fr, y0)

            # Manual reset: re-home the solver, then solve this frame from neutral.
            # prev_q is dropped too, so the causal clamp does not drag the recovered
            # pose back toward the bent one it was just asked to abandon.
            if _reset_requested:
                _reset_requested = False
                rt.configuration.update(q_home)
                prev_q = None
                stuck_n = 0
                print("[gmr_stream] reset: IK re-homed to the neutral pose", file=sys.stderr)

            q = rt.retarget(fr, offset_to_ground=not a.no_ground)

            # Automatic escape from a bent local minimum.  The warm start is what
            # gets stuck, so re-solve the same targets from neutral -- but keep that
            # result only if it genuinely scores better, because from a cold start
            # the IK can also land far worse than a healthy warm solve.
            if not a.no_reset_on_stuck:
                err = ik_residual(rt)
                stuck_n = stuck_n + 1 if err > a.reset_stuck_err else 0
                if stuck_n >= a.reset_stuck_frames:
                    stuck_n = 0
                    q_warm = rt.configuration.q.copy()
                    rt.configuration.update(q_home)
                    q_alt = rt.retarget(fr, offset_to_ground=not a.no_ground)
                    err_alt = ik_residual(rt)
                    if err_alt < err:
                        q = q_alt
                        nrehome += 1
                        print(f"[gmr_stream] stuck (residual {err:.2f}) -> re-homed, "
                              f"residual {err_alt:.2f}", file=sys.stderr)
                    else:
                        rt.configuration.update(q_warm)   # escape was worse; keep the warm solve

            if not a.no_despike and prev_q is not None:
                q, clamped = causal_clamp(prev_q, q, a.despike_root_deg,
                                          a.despike_pos_m, a.despike_dof_deg)
                nclamp += int(clamped)
            prev_q = q.copy()

            sink.step(q, copy.deepcopy(rt.scaled_human_data))
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()
        if sink is not None:
            sink.close()
        if nclamp:
            print(f"[gmr_stream] causal despike clamped {nclamp} frame(s)", file=sys.stderr)
        if nrehome:
            print(f"[gmr_stream] re-homed out of a stuck pose {nrehome} time(s)", file=sys.stderr)
        if nidle:
            print(f"[gmr_stream] eased to the neutral pose {nidle} time(s)", file=sys.stderr)

if __name__ == "__main__":
    main()
