#!/usr/bin/env python3
"""
gen_mixamo_bvh.py — generate a Mixamo ("mixamorig:") T-pose BVH template.

This is the Mixamo analogue of tools/gen_mhr_bvh.py: it emits a skeleton-only
BVH (one neutral rest frame) that src/bvh_writer.cpp can use as a --bvh-template
target.  The MHR network deltas are baked onto whatever rest pose this template
defines, so the only things that must be exactly right here are:

  1. The joint NAMES  — must match the "mixamorig:*" entries in NAME_MAP
                        (src/bvh_writer.cpp).
  2. The rest DIRECTIONS — a clean T-pose (arms along ±X, spine +Y, legs -Y) so
                        the writer's q_bone_align rest-retarget is well-defined.
  3. The CHANNEL ORDER — the writer decomposes non-root joints in ZXY and the
                        root in ZYX, so every joint here is declared
                        "Zrotation Xrotation Yrotation" and the root is
                        "... Zrotation Yrotation Xrotation".  Do NOT reorder.

Bone LENGTHS are only seed values: with the default --bvh-*-shape-change the
writer rewrites every OFFSET to the per-clip median observed length, so the
numbers below just need to be plausible T-pose proportions (centimetres, Y-up).

If you want a *pixel-accurate* Mixamo rest pose instead of these canonical
proportions, export any Mixamo character (T-pose) to BVH from Blender with
rotation order ZXY and use that file directly — the NAME_MAP will pick it up.

Usage:
    python3 tools/gen_mixamo_bvh.py [out=mixamo.bvh]
"""

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'mixamo.bvh')

# ── Skeleton definition ───────────────────────────────────────────────────────
# Each node: (name, offset, children, end_offset_or_None)
# Offsets are LOCAL (child relative to parent), centimetres, Y-up, left = +X.
# The whole left arm/leg/hand subtree is authored once and mirrored to the right
# (X negated) by mirror() below, so the two sides stay symmetric by construction.

def J(name, off, kids=None, end=None):
    return {'name': name, 'off': off, 'kids': kids or [], 'end': end}

def finger(prefix, base, seg, n=3):
    """A finger chain prefix+'1..n' with an End Site; base = knuckle offset."""
    # Build innermost-out so each segment nests in the previous one.
    node = J(prefix + str(n), [seg, 0, 0], end=[seg * 0.7, 0, 0])
    for i in range(n - 1, 0, -1):
        node = J(prefix + str(i), [seg, 0, 0] if i > 1 else base, [node])
    return node

def hand(side):  # side = 'Left' / 'Right'
    p = 'mixamorig:%sHand' % side
    # knuckle spread across the palm (Z), fingers run out along +X
    return [
        J(p + 'Thumb1', [3, -1, 3], [
            J(p + 'Thumb2', [3, 0, 2], [
                J(p + 'Thumb3', [3, 0, 1], end=[2, 0, 1])])]),
        finger(p + 'Index',  [10, 0,  3], 4),
        finger(p + 'Middle', [10, 0,  1], 4),
        finger(p + 'Ring',   [10, 0, -1], 4),
        finger(p + 'Pinky',  [ 9, 0, -3], 3),
    ]

def arm(side):
    s = 'mixamorig:%s' % side
    return J(s + 'Shoulder', [6, 12, -2], [
        J(s + 'Arm', [13, 0, 0], [
            J(s + 'ForeArm', [28, 0, 0], [
                J(s + 'Hand', [26, 0, 0], hand(side))])])])

def leg(side):
    s = 'mixamorig:%s' % side
    return J(s + 'UpLeg', [9, -6, 0], [
        J(s + 'Leg', [0, -42, 0], [
            J(s + 'Foot', [0, -42, 0], [
                J(s + 'ToeBase', [0, -7, 13], end=[0, 0, 7])])])])

def mirror(node):
    """Deep-copy a Left* subtree into its Right* mirror (names + X negated)."""
    nm  = node['name'].replace('Left', 'Right')
    off = [-node['off'][0], node['off'][1], node['off'][2]]
    end = None if node['end'] is None else [-node['end'][0], node['end'][1], node['end'][2]]
    return J(nm, off, [mirror(k) for k in node['kids']], end)

la, ra = arm('Left'), mirror(arm('Left'))
ll, rl = leg('Left'), mirror(leg('Left'))

SKELETON = J('mixamorig:Hips', [0, 0, 0], [
    J('mixamorig:Spine', [0, 10, 0], [
        J('mixamorig:Spine1', [0, 12, 0], [
            J('mixamorig:Spine2', [0, 13, 0], [
                J('mixamorig:Neck', [0, 17, 0], [
                    J('mixamorig:Head', [0, 10, 0], end=[0, 18, 0])]),
                la, ra])])]),
    ll, rl,
])

# ── Emit ──────────────────────────────────────────────────────────────────────
# Channel-order contract with src/bvh_writer.cpp (do not change):
ROOT_CH  = 'Xposition Yposition Zposition Zrotation Yrotation Xrotation'
JOINT_CH = 'Zrotation Xrotation Yrotation'

def fmt(v): return ' '.join('%.6f' % x for x in v)

def emit(node, depth, is_root, out):
    pad = '  ' * depth
    kind = 'ROOT' if is_root else 'JOINT'
    out.append('%s%s %s' % (pad, kind, node['name']))
    out.append('%s{' % pad)
    out.append('%s  OFFSET %s' % (pad, fmt(node['off'])))
    out.append('%s  CHANNELS %s' % (pad, ('6 ' + ROOT_CH) if is_root else ('3 ' + JOINT_CH)))
    for k in node['kids']:
        emit(k, depth + 1, False, out)
    if not node['kids']:
        end = node['end'] if node['end'] is not None else [0, 0, 0]
        out.append('%s  End Site' % pad)
        out.append('%s  {' % pad)
        out.append('%s    OFFSET %s' % (pad, fmt(end)))
        out.append('%s  }' % pad)
    out.append('%s}' % pad)

def count_channels(node, is_root):
    n = 6 if is_root else 3
    for k in node['kids']:
        n += count_channels(k, False)
    return n

lines = ['HIERARCHY']
emit(SKELETON, 0, True, lines)
nch = count_channels(SKELETON, True)
lines += ['MOTION', 'Frames: 1', 'Frame Time: 0.033333', fmt([0.0] * nch)]

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('[gen_mixamo_bvh] wrote %s  (%d channels/frame)' % (OUT, nch))
