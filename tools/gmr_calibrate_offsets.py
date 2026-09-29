#!/usr/bin/env python3
"""
gmr_calibrate_offsets.py — calibrate GMR rotation offsets against GROUND TRUTH.

The position-only retarget can't pin orientation, and geometric/guessed offsets put
the arms in the wrong robot-link frame (the right arm flips +-180 deg).  This tool
solves the offsets empirically against a *known-correct* robot motion:

  reference LAFAN1 clip  --stock GMR-->  CORRECT robot link orientations   (ground truth)
                          --render-->    clean robot video
                          --extract-->   our MHR/lafan_mhr capture of the same motion
  offset[link] = average_t( our_joint_quat(t)^-1 . correct_robot_link_quat(t) )

Because the robot link frames are the ground truth, the resulting offsets are in the
*correct* frame (no +-180 ambiguity), including the left/right handedness — which
guessing the sagittal mirror could not get right.

Two modes (the middle step is an external C++ extraction, so it's split):

  1) RENDER + dump ground-truth orientations:
       GMR/venv/bin/python tools/gmr_calibrate_offsets.py --render \
           --reference scripts/gmr_configs/reference_lafan1_dance1_subject2.bvh \
           --out /tmp/cal --frames 600
     -> writes /tmp/cal/clean.mp4 and /tmp/cal/robot_orient.npz

  2) (extract the clean video with our pipeline, lafan_mhr template)
       scripts/offline_video.sh --from /tmp/cal/clean.mp4 --bvh /tmp/cal/clean.bvh \
           --bvh-template bvh/lafan_mhr.bvh --interpolate-jitter
     -> /tmp/cal/clean_0.bvh   (use the longest track)

  3) SOLVE the offsets and print them (auto-scans small frame shifts for alignment):
       GMR/venv/bin/python tools/gmr_calibrate_offsets.py --solve \
           --capture /tmp/cal/clean_0.bvh --orient /tmp/cal/robot_orient.npz

The clean render uses follow_camera (fixed angle, re-centres only) and NO human
overlay, so the capture is a single clean track aligned 1:1 with the reference.
Note: SAM3DBody fits a *human* to the G1's thin arms, so the per-frame fit is noisy
(offset dev ~15-25 deg) — the offset MEAN is still in the correct frame.  Average
over a longer clip for sharper offsets.
"""
import argparse, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# robot links we calibrate (and the human joints that drive them)
MAP = {"left_shoulder_yaw_link":"LeftArm","left_elbow_link":"LeftForeArm","left_wrist_yaw_link":"LeftHand",
       "right_shoulder_yaw_link":"RightArm","right_elbow_link":"RightForeArm","right_wrist_yaw_link":"RightHand"}

def do_render(a):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames, _ = load_bvh_file(a.reference, format="lafan1")
    N = min(a.frames, len(frames))
    rt = GMR(src_human="bvh_lafan1", tgt_robot=a.robot, actual_human_height=1.75, verbose=False)
    vw = RobotMotionViewer(robot_type=a.robot, motion_fps=30, record_video=True, video_path=str(out/"clean.mp4"))
    links = list(MAP) + ["pelvis"]
    rec = {l: [] for l in links}
    for i in range(N):
        q = rt.retarget(frames[i])
        vw.step(root_pos=q[:3], root_rot=q[3:7], dof_pos=q[7:], human_motion_data=None,
                rate_limit=False, follow_camera=True)
        for l in links:
            m = rt.configuration.data.xmat[rt.robot_body_names[l]].reshape(3,3)
            rec[l].append(R.from_matrix(m).as_quat())
    vw.close()
    np.savez(str(out/"robot_orient.npz"), N=N, **{l: np.array(rec[l]) for l in links})
    print(f"wrote {out}/clean.mp4 and robot_orient.npz (N={N})")
    print("next: extract clean.mp4 with --bvh-template bvh/lafan_mhr.bvh, then run --solve")

def do_solve(a):
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    cap, _ = load_bvh_file(a.capture, format="lafan1")
    D = np.load(a.orient); N = min(len(cap), int(D["N"]))
    def ourR(fr, j): q = fr[j][1]; return R.from_quat([q[1],q[2],q[3],q[0]])
    def offs_at(shift):
        res = {}
        for rl, hj in MAP.items():
            qs = []
            for i in range(N):
                j = i + shift
                if 0 <= j < D[rl].shape[0] and i < len(cap):
                    qs.append((ourR(cap[i],hj).inv()*R.from_quat(D[rl][j])).as_quat())
            qs = np.array(qs); qs[(qs@qs[0])<0] *= -1; m = qs.mean(0); m /= np.linalg.norm(m)
            dev = np.array([(R.from_quat(x*np.sign(x@m)).inv()*R.from_quat(m)).magnitude() for x in qs])*180/np.pi
            res[rl] = (m, dev.mean())
        return res
    best = min(((np.mean([v[1] for v in offs_at(s).values()]), s) for s in range(-8,12)))
    r = offs_at(best[1])
    print(f"best frame shift={best[1]}  mean dev={best[0]:.1f}deg\n")
    print(f"{'robot_link':24s} rot_offset (wxyz)                          dev")
    for rl in MAP:
        m, d = r[rl]
        q = [round(float(m[3]),4), round(float(m[0]),4), round(float(m[1]),4), round(float(m[2]),4)]
        print(f"{rl:24s} {str(q):42s} {d:5.1f}")

ap = argparse.ArgumentParser()
ap.add_argument("--render", action="store_true")
ap.add_argument("--solve", action="store_true")
ap.add_argument("--robot", default="unitree_g1")
ap.add_argument("--reference", help="known-good real LAFAN1 BVH (for --render)")
ap.add_argument("--frames", type=int, default=600)
ap.add_argument("--out", default="/tmp/cal")
ap.add_argument("--capture", help="our lafan_mhr capture of the clean render (for --solve)")
ap.add_argument("--orient", help="robot_orient.npz from --render (for --solve)")
a = ap.parse_args()
if a.render: do_render(a)
elif a.solve: do_solve(a)
else: ap.error("pass --render or --solve")
