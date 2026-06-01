#!/usr/bin/env python3
"""
check_mesh_bvh_overlay.py — verify that an --export-mesh OBJ file overlays its
matching --bvh file correctly when both are imported into Blender.

Usage:
    python3 tools/check_mesh_bvh_overlay.py  MESH.obj  SKELETON.bvh

The script reads the BVH root (hip) position from frame 0 and finds the nearest
OBJ vertex.  A well-aligned export shows:
  • nearest vertex within ~15 cm of the BVH root  (it's a joint centre vs mesh)
  • OBJ Y range spanning roughly [0-30, 150-200] cm  (feet near floor, head above)
  • BVH root Y ≈ 90-120 cm  (hip height above floor for a standing adult)

Blender import recipe (no rotation needed for the corrected export):
  1. File → Import → Wavefront (.obj) → pick MESH.obj  → Import
  2. File → Import → Motion Capture (.bvh) → pick SKELETON.bvh → Import
  Both objects share world Y-up space: the skeleton should overlay the mesh.
"""

import sys
import math
import struct

# ── helpers ──────────────────────────────────────────────────────────────────

def load_obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return verts


def bvh_root_frame0(path):
    """Return (X, Y, Z) cm of the BVH root joint at frame 0."""
    with open(path) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip() == 'MOTION':
            vals = [float(x) for x in lines[i + 3].split()]
            return (vals[0], vals[1], vals[2])
    raise RuntimeError("MOTION section not found in " + path)


def bvh_root_range(path):
    """Return (Y_min, Y_max) cm of the BVH root across all frames."""
    with open(path) as f:
        lines = f.readlines()
    ys = []
    in_motion = False
    frame_time_seen = False
    for l in lines:
        if l.strip() == 'MOTION':
            in_motion = True
            continue
        if in_motion and l.startswith('Frame Time'):
            frame_time_seen = True
            continue
        if in_motion and frame_time_seen:
            vals = l.split()
            if len(vals) >= 3:
                ys.append(float(vals[1]))
    return (min(ys), max(ys)) if ys else (0.0, 0.0)


def nearest_vertex(verts, target):
    best_d, best_v, best_i = 1e9, None, 0
    for i, v in enumerate(verts):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, target)))
        if d < best_d:
            best_d, best_v, best_i = d, v, i
    return best_i, best_v, best_d


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: check_mesh_bvh_overlay.py MESH.obj SKELETON.bvh")
        sys.exit(1)

    obj_path = sys.argv[1]
    bvh_path = sys.argv[2]

    print(f"OBJ : {obj_path}")
    print(f"BVH : {bvh_path}")
    print()

    verts = load_obj_vertices(obj_path)
    root = bvh_root_frame0(bvh_path)

    print(f"BVH root frame-0  : X={root[0]:.1f}  Y={root[1]:.1f}  Z={root[2]:.1f} cm")
    print(f"OBJ vertex count  : {len(verts)}")

    ys = [v[1] for v in verts]
    y_min, y_max = min(ys), max(ys)
    y_centroid = sum(ys) / len(ys)
    print(f"OBJ Y range       : [{y_min:.1f}, {y_max:.1f}] cm  (height span = {y_max - y_min:.1f} cm)")
    print(f"OBJ Y centroid    : {y_centroid:.1f} cm")
    print()

    idx, v, dist = nearest_vertex(verts, root)
    print(f"Nearest OBJ vertex to BVH root:")
    print(f"  index  = {idx}")
    print(f"  pos    = ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) cm")
    print(f"  dist   = {dist:.2f} cm  (< 15 cm = good alignment)")
    print()

    y_lo, y_hi = bvh_root_range(bvh_path)
    print(f"BVH root Y range across all frames: [{y_lo:.1f}, {y_hi:.1f}] cm")
    print()

    # ── Verdict ──────────────────────────────────────────────────────────────
    ok = True
    if dist > 20:
        print("WARN  Nearest vertex is > 20 cm from BVH root — possible coordinate mismatch.")
        ok = False
    if y_max < 100:
        print("WARN  OBJ max Y < 100 cm — mesh may not be in world Y-up space yet.")
        ok = False
    if y_min > 50:
        print("WARN  OBJ min Y > 50 cm — head/upper body may be missing or misoriented.")
        ok = False
    if root[1] < 70 or root[1] > 140:
        print(f"WARN  BVH root Y = {root[1]:.1f} cm — outside typical adult hip range [70, 140].")
        ok = False

    if ok:
        print("OK    OBJ and BVH appear well-aligned.  Load both in Blender with default")
        print("      import settings (no extra rotation) and they should overlay correctly.")
    else:
        print("      Re-export with --export-mesh using the corrected build to fix alignment.")


if __name__ == "__main__":
    main()
