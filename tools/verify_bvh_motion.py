#!/usr/bin/env python3
"""Forward-kinematics a BVH and compare to an MHR keypoints CSV.

The MHR CSV has world-space joint positions per frame (the `--out` output of
`fast_sam_3dbody_run`).  We FK the BVH ourselves (so we don't have to fight
BVHTester's behind-camera filter), then for a small set of corresponding
joints we report:

   • per-frame Euclidean error after subtracting the root (hip) so absolute
     translation cancels;
   • the median, p90, and max over the whole video.

Usage:
    python3 tools/verify_bvh_motion.py /tmp/boom.bvh /tmp/boom_mhr.csv
"""
from __future__ import annotations
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── BVH parser + FK ────────────────────────────────────────────────────────

@dataclass
class BvhJoint:
    name: str
    parent: int
    offset: np.ndarray  # (3,)
    channels: List[str] = field(default_factory=list)
    channel_start: int = -1


def parse_bvh(path: Path):
    text = path.read_text()
    tokens = text.split()
    joints: List[BvhJoint] = []
    stack: List[int] = []
    pending: Optional[str] = None
    channel_cursor = 0
    motion_idx = None

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "MOTION":
            motion_idx = i
            break
        elif t in ("ROOT", "JOINT"):
            pending = tokens[i + 1]
            i += 2
            continue
        elif t == "End" and i + 1 < len(tokens) and tokens[i + 1] == "Site":
            pending = None
            i += 2
            continue
        elif t == "{":
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(name=pending or "", parent=parent, offset=np.zeros(3)))
            stack.append(len(joints) - 1)
            pending = None
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            j = stack[-1]
            joints[j].offset = np.array([float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            j = stack[-1]
            joints[j].channels = tokens[i + 2 : i + 2 + n]
            joints[j].channel_start = channel_cursor
            channel_cursor += n
            i += 2 + n
        else:
            i += 1

    # Parse the MOTION block by re-reading the text from "MOTION" onwards
    # (text-based parsing is far less error-prone than re-using whitespace
    # tokens for the header lines).
    motion_text = text[text.find("MOTION") + len("MOTION"):]
    lines = [l.strip() for l in motion_text.splitlines() if l.strip()]
    assert lines[0].startswith("Frames:"), lines[0]
    n_frames = int(lines[0].split(":")[1].strip())
    assert lines[1].startswith("Frame Time:"), lines[1]
    frame_time = float(lines[1].split(":")[1].strip())
    data_lines = lines[2 : 2 + n_frames]
    rows = [list(map(float, ln.split())) for ln in data_lines]
    frame_data = np.array(rows, dtype=float)
    if frame_data.shape != (n_frames, channel_cursor):
        print(f"warning: motion shape {frame_data.shape}, expected ({n_frames},{channel_cursor})",
              file=sys.stderr)
    return joints, frame_data


# ── Rotation matrices in the BVHTester convention ─────────────────────────
# BVHTester applies M = R_z · R_x · R_y for "Zrotation Xrotation Yrotation",
# and the rotation angles are *negated* before building the matrix (see
# bvh_transform.c).  We follow the same convention here.

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix_from_channels(channels: List[str], values: np.ndarray) -> np.ndarray:
    """Build the joint local rotation matrix.

    AmMatrix (which drives BVHTester) stores rotation matrices as the
    transpose of the standard right-handed form AND negates the angles
    before composing.  Net effect: it computes M = Rz_std(θz)·Rx_std(θx)·
    Ry_std(θy) with standard right-handed matrices and *positive* angles.
    Replicate that here with no negation.
    """
    m = np.eye(3)
    for ch, v in zip(channels, values):
        a = math.radians(v)
        if ch == "Xrotation":
            m = m @ Rx(a)
        elif ch == "Yrotation":
            m = m @ Ry(a)
        elif ch == "Zrotation":
            m = m @ Rz(a)
    return m


def fk_world_positions(joints, frame_row) -> np.ndarray:
    """Return [n_joints, 3] world-space positions for one BVH frame."""
    pos = np.zeros((len(joints), 3))
    rot = [np.eye(3)] * len(joints)
    for i, j in enumerate(joints):
        # local translation: OFFSET + any positional channels
        t = j.offset.copy()
        if j.channels and j.channels[0].endswith("position"):
            # root: first 3 channels are X/Y/Zposition
            t = np.array([
                frame_row[j.channel_start + 0],
                frame_row[j.channel_start + 1],
                frame_row[j.channel_start + 2],
            ])
            rot_channels = j.channels[3:]
            rot_values = frame_row[j.channel_start + 3 : j.channel_start + 6]
        else:
            rot_channels = j.channels
            rot_values = frame_row[j.channel_start : j.channel_start + len(j.channels)]
        R_local = matrix_from_channels(rot_channels, rot_values)
        if j.parent < 0:
            rot[i] = R_local
            pos[i] = t
        else:
            rot[i] = rot[j.parent] @ R_local
            pos[i] = pos[j.parent] + rot[j.parent] @ t
    return pos


# ── MHR CSV loader ─────────────────────────────────────────────────────────

# The MHR-70 keypoint names that overlap with the BVH joints we care about.
BVH_TO_MHR_KP = {
    "hip":        ("left_hip", "right_hip"),     # midpoint
    "neck":       ("neck",),
    "head":       ("nose",),
    "lShldr":     ("left_shoulder",),
    "rShldr":     ("right_shoulder",),
    "lForeArm":   ("left_elbow",),
    "rForeArm":   ("right_elbow",),
    "lHand":      ("left_wrist",),
    "rHand":      ("right_wrist",),
    "lThigh":     ("left_hip",),
    "rThigh":     ("right_hip",),
    "lShin":      ("left_knee",),
    "rShin":      ("right_knee",),
    "lFoot":      ("left_ankle",),
    "rFoot":      ("right_ankle",),
}


def load_mhr_csv(path: Path):
    """Return dict frame -> dict joint_name -> (x,y,z)."""
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        # columns:  frame, skeleton_id, name_x, name_y, name_z, ...
        names = []
        for i in range(2, len(header), 3):
            base = header[i][:-2]   # strip "_x"
            names.append(base)
        frames: Dict[int, Dict[str, np.ndarray]] = {}
        for row in reader:
            if not row: continue
            fr = int(row[0])
            sk = int(row[1])
            if sk != 0: continue
            d: Dict[str, np.ndarray] = {}
            for k, name in enumerate(names):
                x, y, z = (float(row[2 + 3*k]), float(row[3 + 3*k]), float(row[4 + 3*k]))
                d[name] = np.array([x, y, z])
            frames[fr] = d
    return frames


# ── Comparison ────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 3:
        print("usage: verify_bvh_motion.py BVH_FILE MHR_CSV", file=sys.stderr)
        return 1
    bvh_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    joints, motion = parse_bvh(bvh_path)
    print(f"BVH: {len(joints)} joints, {motion.shape[0]} frames, {motion.shape[1]} channels")
    name_to_idx = {j.name: i for i, j in enumerate(joints)}

    mhr_frames = load_mhr_csv(csv_path)
    print(f"MHR CSV: {len(mhr_frames)} frames")

    # MHR positions are in metres in the camera frame.  Multiply by 100 to be
    # in cm (BVH units), and then compare hip-relative.
    pairs = [(b, m) for b, m in BVH_TO_MHR_KP.items() if b in name_to_idx]
    print(f"Comparing {len(pairs)} joints")

    # Pick a sample of frames spread across the video.
    n_frames = min(motion.shape[0], len(mhr_frames))
    sample_idx = list(range(0, n_frames, max(1, n_frames // 40)))[:40]

    per_joint_errors = {b: [] for b, _ in pairs}
    for fi in sample_idx:
        if fi not in mhr_frames:
            continue
        kps = mhr_frames[fi]
        # MHR midhip (for hip BVH joint)
        if "left_hip" not in kps or "right_hip" not in kps:
            continue
        mhr_hip = 0.5 * (kps["left_hip"] + kps["right_hip"]) * 100.0  # m → cm

        bvh_world = fk_world_positions(joints, motion[fi])
        bvh_hip = bvh_world[name_to_idx["hip"]]

        for bvh_name, mhr_names in pairs:
            mhr_pos = None
            if len(mhr_names) == 1:
                if mhr_names[0] in kps:
                    mhr_pos = kps[mhr_names[0]] * 100.0
            elif len(mhr_names) == 2:
                if mhr_names[0] in kps and mhr_names[1] in kps:
                    mhr_pos = 0.5 * (kps[mhr_names[0]] + kps[mhr_names[1]]) * 100.0
            if mhr_pos is None: continue

            bvh_p = bvh_world[name_to_idx[bvh_name]]

            # hip-relative residual.  MHR cam_y points down, MHR z points away
            # from camera.  Flip both so the BVH (Y up, Z toward camera) and
            # MHR live in the same frame for the comparison.
            mhr_rel = mhr_pos - mhr_hip
            mhr_rel = np.array([mhr_rel[0], -mhr_rel[1], -mhr_rel[2]])
            bvh_rel = bvh_p - bvh_hip
            err = np.linalg.norm(bvh_rel - mhr_rel)
            per_joint_errors[bvh_name].append(err)

    print(f"\n  {'joint':<10}  {'frames':>6}  {'median':>8}  {'p90':>8}  {'max':>8}  (cm)")
    print("  " + "-" * 60)
    for name in [b for b, _ in pairs]:
        errs = per_joint_errors[name]
        if not errs: continue
        a = np.array(errs)
        print(f"  {name:<10}  {len(errs):>6}  {np.median(a):>8.2f}  {np.percentile(a,90):>8.2f}  {a.max():>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
