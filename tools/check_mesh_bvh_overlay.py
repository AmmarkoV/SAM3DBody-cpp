#!/usr/bin/env python3
"""
check_mesh_bvh_overlay.py — verify that an --export-mesh OBJ overlays its
matching --bvh file correctly when both are imported into Blender.

Usage:
    python3 tools/check_mesh_bvh_overlay.py  MESH.obj  SKELETON.bvh  [FRAME]

    FRAME defaults to 0 (first frame).

Reports:
  • Overall position check (pelvis BVH root vs nearest OBJ vertex)
  • Per-joint alignment: BVH FK world position vs nearest OBJ vertex
  • Histogram of OBJ vertex Y values (sanity-checks body orientation)

Blender import recipe (no rotation needed for the corrected export):
  1. File → Import → Wavefront (.obj) → pick MESH.obj  → Import
  2. File → Import → Motion Capture (.bvh) → pick SKELETON.bvh → Import
  Both use world Y-up space; the skeleton should overlay the mesh.
"""

import sys, math, struct, re
from collections import defaultdict


# ── BVH parser ────────────────────────────────────────────────────────────────

class BVHJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent          # BVHJoint or None
        self.children = []
        self.offset = [0.0, 0.0, 0.0]
        self.channels = []            # list of channel name strings


def parse_bvh(path):
    """Return (root_joint, frames, n_frames, frame_time)."""
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)

    def nxt(): return next(it)

    def parse_joint(parent=None):
        name = nxt()
        joint = BVHJoint(name, parent)
        assert nxt() == '{'
        while True:
            tok = nxt()
            if tok == 'OFFSET':
                joint.offset = [float(nxt()), float(nxt()), float(nxt())]
            elif tok == 'CHANNELS':
                n = int(nxt())
                joint.channels = [nxt() for _ in range(n)]
            elif tok == 'JOINT':
                child = parse_joint(joint)
                joint.children.append(child)
            elif tok == 'End':
                nxt()   # 'Site'
                nxt()   # '{'
                nxt()   # 'OFFSET'
                nxt(); nxt(); nxt()
                nxt()   # '}'
            elif tok == '}':
                break
        return joint

    root = None
    frames = []
    n_frames = 0
    frame_time = 1/30

    while True:
        try:
            tok = nxt()
        except StopIteration:
            break
        if tok == 'ROOT':
            root = parse_joint()
        elif tok == 'MOTION':
            nxt()                              # 'Frames:'
            n_frames = int(nxt())
            nxt(); nxt()                       # 'Frame' 'Time:'
            frame_time = float(nxt())
            for _ in range(n_frames):
                row = []
                # Collect all channels: count them from root
                def count_ch(j):
                    return sum(len(c.channels) for c in _all_joints(j))
                if not frames:
                    ch_total = count_ch(root) if root else 0
                row = [float(nxt()) for _ in range(ch_total or count_ch(root))]
                frames.append(row)

    return root, frames, n_frames, frame_time


def _all_joints(j):
    yield j
    for c in j.children:
        yield from _all_joints(c)


# ── BVH Forward Kinematics ────────────────────────────────────────────────────

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1,0,0],[0,c,-s],[0,s,c]]

def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c,0,s],[0,1,0],[-s,0,c]]

def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c,-s,0],[s,c,0],[0,0,1]]

def _mul33(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _mrot(m, v):
    return [sum(m[i][j]*v[j] for j in range(3)) for i in range(3)]

def _add3(a, b): return [a[i]+b[i] for i in range(3)]


def bvh_fk(root, frame_data, frame_idx=0):
    """Return dict joint_name → world_position (cm) for the given frame."""
    row = frame_data[frame_idx]
    ch_iter = iter(row)
    positions = {}

    def process(joint, parent_pos, parent_rot):
        chs = {ch: next(ch_iter) for ch in joint.channels}
        pos = list(parent_pos)
        rot = [list(r) for r in parent_rot]

        # Translation channels
        local_off = list(joint.offset)
        if 'Xposition' in chs: local_off[0] = chs['Xposition']
        if 'Yposition' in chs: local_off[1] = chs['Yposition']
        if 'Zposition' in chs: local_off[2] = chs['Zposition']

        world_off = _mrot(parent_rot, local_off)
        pos = _add3(pos, world_off)

        # Rotation channels (applied in the order listed)
        R = [[1,0,0],[0,1,0],[0,0,1]]
        for ch in joint.channels:
            if ch not in chs: continue
            val = math.radians(chs[ch])
            if   ch == 'Xrotation': R = _mul33(R, _rx(val))
            elif ch == 'Yrotation': R = _mul33(R, _ry(val))
            elif ch == 'Zrotation': R = _mul33(R, _rz(val))
        rot = _mul33(parent_rot, R)

        positions[joint.name] = pos
        for child in joint.children:
            process(child, pos, rot)

    process(root, [0.0, 0.0, 0.0], [[1,0,0],[0,1,0],[0,0,1]])
    return positions


# ── OBJ loader ───────────────────────────────────────────────────────────────

def load_obj_vertices(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return verts


def nearest_vertex(verts, target):
    best_d, best_v, best_i = 1e9, None, 0
    tx, ty, tz = target
    for i, (x, y, z) in enumerate(verts):
        d = math.sqrt((x-tx)**2+(y-ty)**2+(z-tz)**2)
        if d < best_d:
            best_d, best_v, best_i = d, (x,y,z), i
    return best_i, best_v, best_d


# ── Main ─────────────────────────────────────────────────────────────────────

# Body joints we care about for alignment checks (BVH names from body.bvh)
BODY_JOINTS = [
    'hip', 'abdomen', 'chest', 'neck', 'head',
    'lCollar', 'lShldr', 'lForeArm', 'lHand',
    'rCollar', 'rShldr', 'rForeArm', 'rHand',
    'lThigh', 'lShin', 'lFoot',
    'rThigh', 'rShin', 'rFoot',
]

# BVH joint name → MHR joint name, for centre-to-centre comparison against the
# companion .joints dump (mirrors NAME_MAP in src/bvh_writer.cpp).
BVH_TO_MHR = {
    'hip':'root','abdomen':'c_spine1','chest':'c_spine3','neck':'c_neck','head':'c_head',
    'lCollar':'l_clavicle','lShldr':'l_uparm','lForeArm':'l_lowarm','lHand':'l_wrist',
    'rCollar':'r_clavicle','rShldr':'r_uparm','rForeArm':'r_lowarm','rHand':'r_wrist',
    'lThigh':'l_upleg','lShin':'l_lowleg','lFoot':'l_foot',
    'rThigh':'r_upleg','rShin':'r_lowleg','rFoot':'r_foot',
}

def load_mhr_joints(path):
    """Read a PREFIX_pP_FFFFF.joints file → {idx: (x,y,z)}.  None if missing."""
    import os
    if not os.path.exists(path): return None
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                out[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    return out

def mhr_name_to_idx():
    import re, os
    # Locate mhr_joint_table.h relative to this tool (../src/).
    here = os.path.dirname(os.path.abspath(__file__))
    hdr = os.path.join(here, '..', 'src', 'mhr_joint_table.h')
    if not os.path.exists(hdr): return {}
    block = open(hdr).read().split('NAMES[N_JOINTS] = {',1)[1].split('};',1)[0]
    names = re.findall(r'"([^"]+)"', block)
    return {n:i for i,n in enumerate(names)}


def main():
    if len(sys.argv) < 3:
        print("Usage: check_mesh_bvh_overlay.py MESH.obj SKELETON.bvh [FRAME]")
        sys.exit(1)

    obj_path  = sys.argv[1]
    bvh_path  = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"OBJ   : {obj_path}")
    print(f"BVH   : {bvh_path}")
    print(f"Frame : {frame_idx}")
    print()

    verts = load_obj_vertices(obj_path)
    root, frames, n_frames, frame_time = parse_bvh(bvh_path)
    if frame_idx >= n_frames:
        print(f"WARN: frame {frame_idx} >= n_frames {n_frames}; using last frame.")
        frame_idx = n_frames - 1

    joint_world = bvh_fk(root, frames, frame_idx)

    ys = [v[1] for v in verts]
    print(f"OBJ : {len(verts)} vertices,  Y=[{min(ys):.0f}, {max(ys):.0f}] cm  "
          f"(span {max(ys)-min(ys):.0f} cm)")
    print(f"BVH : {n_frames} frames,  root Y={joint_world.get('hip',[0,0,0])[1]:.1f} cm")
    print()

    # ── Y histogram ──────────────────────────────────────────────────────────
    print("OBJ Y histogram (each '#' ≈ 50 vertices):")
    step = 20
    bins = defaultdict(int)
    for y in ys: bins[int(y // step) * step] += 1
    for b in sorted(bins):
        bar = '#' * (bins[b] // 50)
        print(f"  [{b:+4d},{b+step:+4d}) {bins[b]:5d} {bar}")
    print()

    # ── Per-joint alignment ───────────────────────────────────────────────────
    print(f"{'Joint':12s}  {'BVH world pos (cm)':28s}  {'Nearest vertex':28s}  {'dist cm':>7s}  status")
    print('-'*90)

    any_warn = False
    WARN_THR = 15.0   # cm; further than this is suspicious

    for jname in BODY_JOINTS:
        if jname not in joint_world:
            continue
        bpos = joint_world[jname]
        idx, vpos, dist = nearest_vertex(verts, bpos)
        status = 'OK' if dist < WARN_THR else 'WARN'
        if status == 'WARN': any_warn = True
        print(f"{jname:12s}  ({bpos[0]:+7.1f},{bpos[1]:+7.1f},{bpos[2]:+7.1f})  "
              f"({vpos[0]:+7.1f},{vpos[1]:+7.1f},{vpos[2]:+7.1f})  "
              f"{dist:7.1f}  {status}")

    print()
    if any_warn:
        print("WARN  Some joints are > 15 cm from the nearest mesh vertex.")
        print("      Note: a joint CENTRE sits inside the body, so a few cm of this is the")
        print("      joint-to-surface gap, not skeleton error.  See the centre-to-centre")
        print("      table below (if a .joints file was exported) for the true skeleton error.")
    else:
        print("OK    All body joints are within 15 cm of the mesh surface.  Load both in")
        print("      Blender with default import settings; they should overlay correctly.")

    # ── Centre-to-centre: BVH joint vs MHR joint (true skeleton error) ─────────
    # The .obj's companion .joints file holds the MHR joint CENTRES in the same
    # world space.  Comparing BVH-FK joints to those removes the joint-to-surface
    # offset, isolating skeleton/OFFSET error from the rotation retarget.
    jpath = obj_path[:-4] + '.joints' if obj_path.endswith('.obj') else obj_path + '.joints'
    mhrj = load_mhr_joints(jpath)
    if mhrj:
        n2i = mhr_name_to_idx()
        print()
        print("Centre-to-centre (BVH joint vs MHR joint — true skeleton error):")
        print(f"{'Joint':12s}  {'BVH pos (cm)':24s}  {'MHR pos (cm)':24s}  {'dist cm':>7s}")
        print('-'*78)
        tot = 0.0; cnt = 0; worst = (0.0, '')
        for jname in BODY_JOINTS:
            if jname not in joint_world: continue
            mn = BVH_TO_MHR.get(jname)
            if mn is None or mn not in n2i or n2i[mn] not in mhrj: continue
            b = joint_world[jname]; mpos = mhrj[n2i[mn]]
            d = math.sqrt(sum((b[i]-mpos[i])**2 for i in range(3)))
            tot += d; cnt += 1
            if d > worst[0]: worst = (d, jname)
            print(f"{jname:12s}  ({b[0]:+7.1f},{b[1]:+7.1f},{b[2]:+7.1f})  "
                  f"({mpos[0]:+7.1f},{mpos[1]:+7.1f},{mpos[2]:+7.1f})  {d:7.1f}")
        if cnt:
            print('-'*78)
            print(f"{'MEAN':12s}  {'':24s}  {'':24s}  {tot/cnt:7.1f}")
            print(f"worst: {worst[1]} at {worst[0]:.1f} cm")
            print()
            print("This is the real skeleton error (rotation + OFFSET length).  If it is")
            print("small but the surface distances above are large, the residual is just")
            print("joint-centre-to-skin.  If THIS is large, the OFFSETs/retarget need work.")
    else:
        print()
        print(f"(no {jpath} — re-export with the current build to enable the centre-to-centre check)")


if __name__ == "__main__":
    main()
