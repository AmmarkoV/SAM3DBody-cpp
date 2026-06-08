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
a = ap.parse_args()

from general_motion_retargeting import params
# Point the bvh_lafan1 -> <robot> config at our custom file (in-memory patch).
params.IK_CONFIG_DICT["bvh_lafan1"][a.robot] = Path(a.config)

from general_motion_retargeting import GeneralMotionRetargeting as GMR, RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file

frames, h = load_bvh_file(a.bvh, format="lafan1")

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
qpos_list = []
for fr in frames:
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
