#!/usr/bin/env python3
"""
gen_mhr_bvh.py — generate body_mhr.bvh from the MHR rest skeleton.

Takes the MocapNET-derived mocapnet.bvh (MakeHuman joint names + hierarchy +
channels) and rewrites every mapped joint's OFFSET so its rest position equals
its corresponding MHR joint's rest position (from onnx/body_model.lbs).  The
result keeps the MakeHuman names (so NAME_MAP / the Blender associations still
work) but its REST POSE matches MHR exactly.

Why: when the template rest pose == the MHR rest pose, the BVH writer's rest
retarget (q_bone_align) becomes identity, the per-person offset rewrite is a
no-op, and the chest→clavicle "branch joint" direction is correct by
construction — so the arm/collar no longer drift from the mesh.  See
bvh-arm-rest-pose-mismatch memory + tools/check_mesh_bvh_overlay.py.

Usage:
    python3 tools/gen_mhr_bvh.py [mocapnet.bvh] [onnx/body_model.lbs] [out=body_mhr.bvh]

Mapped joints land exactly at MHR rest positions; unmapped cosmetic joints
(e.g. neck1, finger nulls) and End Sites keep their template offsets and are
carried along.  Root (hip) stays at the origin.
"""

import sys, os, struct, re, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'mocapnet.bvh')
if not os.path.exists(SRC):                       # fall back to body.bvh pre-rename
    alt = os.path.join(ROOT, 'body.bvh')
    if os.path.exists(alt): SRC = alt
LBS  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'onnx', 'body_model.lbs')
OUT  = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'body_mhr.bvh')

# ── NAME_MAP (BVH name → MHR name), mirrors src/bvh_writer.cpp ─────────────────
NAME_MAP = {
    'hip':'root','abdomen':'c_spine1','chest':'c_spine3','neck':'c_neck','head':'c_head',
    'jaw':'c_jaw',
    'lCollar':'l_clavicle','lShldr':'l_uparm','lForeArm':'l_lowarm','lHand':'l_wrist',
    'rCollar':'r_clavicle','rShldr':'r_uparm','rForeArm':'r_lowarm','rHand':'r_wrist',
    'lThigh':'l_upleg','lShin':'l_lowleg','lFoot':'l_foot',
    'rThigh':'r_upleg','rShin':'r_lowleg','rFoot':'r_foot',
    'lthumb':'l_thumb1','finger1-2.l':'l_thumb2','finger1-3.l':'l_thumb3',
    'finger2-1.l':'l_index1','finger2-2.l':'l_index2','finger2-3.l':'l_index3',
    'finger3-1.l':'l_middle1','finger3-2.l':'l_middle2','finger3-3.l':'l_middle3',
    'finger4-1.l':'l_ring1','finger4-2.l':'l_ring2','finger4-3.l':'l_ring3',
    'finger5-1.l':'l_pinky1','finger5-2.l':'l_pinky2','finger5-3.l':'l_pinky3',
    'rthumb':'r_thumb1','finger1-2.r':'r_thumb2','finger1-3.r':'r_thumb3',
    'finger2-1.r':'r_index1','finger2-2.r':'r_index2','finger2-3.r':'r_index3',
    'finger3-1.r':'r_middle1','finger3-2.r':'r_middle2','finger3-3.r':'r_middle3',
    'finger4-1.r':'r_ring1','finger4-2.r':'r_ring2','finger4-3.r':'r_ring3',
    'finger5-1.r':'r_pinky1','finger5-2.r':'r_pinky2','finger5-3.r':'r_pinky3',
}

# ── quaternion / FK helpers (match bvh_writer.cpp conventions) ────────────────
def qmul(a,b):
    return [b[0]*a[3]+b[3]*a[0]+b[2]*a[1]-b[1]*a[2],
            b[1]*a[3]-b[2]*a[0]+b[3]*a[1]+b[0]*a[2],
            b[2]*a[3]+b[1]*a[0]-b[0]*a[1]+b[3]*a[2],
            b[3]*a[3]-b[0]*a[0]-b[1]*a[1]-b[2]*a[2]]
def qrot(q,v):
    x,y,z,w=q; tx=2*(y*v[2]-z*v[1]); ty=2*(z*v[0]-x*v[2]); tz=2*(x*v[1]-y*v[0])
    return [v[0]+w*tx+(y*tz-z*ty), v[1]+w*ty+(z*tx-x*tz), v[2]+w*tz+(x*ty-y*tx)]

def mhr_rest_world():
    """Return {mhr_name: (x,y,z)} rest-pose world positions in model cm,
    relative to the pelvis (MHR 'root' joint), matching the export frame."""
    d = open(LBS,'rb').read()
    h = struct.unpack_from('8I', d, 0)
    nj, pc = h[2], h[7]; pr = nj*7; o = 32
    o += pr*pc*4
    jo  = struct.unpack_from(f'{nj*3}f', d, o); o += nj*3*4
    pre = struct.unpack_from(f'{nj*4}f', d, o); o += nj*4*4
    par = struct.unpack_from(f'{nj}i',  d, o)
    gq=[[0,0,0,1] for _ in range(nj)]; gt=[[0,0,0] for _ in range(nj)]
    for j in range(nj):
        pq = list(pre[j*4:j*4+4]); off = jo[j*3:j*3+3]; p = par[j]
        if p < 0 or p >= nj:
            gq[j]=pq; gt[j]=list(off)
        else:
            gq[j]=qmul(gq[p],pq); rt=qrot(gq[p],off)
            gt[j]=[gt[p][i]+rt[i] for i in range(3)]
    names = re.findall(r'"([^"]+)"',
        open(os.path.join(ROOT,'src','mhr_joint_table.h')).read()
        .split('NAMES[N_JOINTS] = {',1)[1].split('};',1)[0])
    n2i = {n:i for i,n in enumerate(names)}
    pelvis = gt[n2i['root']]
    return {n: tuple(gt[i][k]-pelvis[k] for k in range(3)) for n,i in n2i.items()}

# ── BVH parse (tree) ──────────────────────────────────────────────────────────
class Node:
    def __init__(s,name,endsite=False):
        s.name=name; s.endsite=endsite; s.offset=[0,0,0]; s.channels=[]; s.kids=[]; s.world=None

def parse(path):
    toks = open(path).read().split(); it=iter(toks); nx=lambda: next(it)
    def pj(endsite=False):
        if endsite: name='__end'
        else: name=nx()
        n=Node(name,endsite); assert nx()=='{'
        while True:
            t=nx()
            if t=='OFFSET': n.offset=[float(nx()),float(nx()),float(nx())]
            elif t=='CHANNELS':
                k=int(nx()); n.channels=[nx() for _ in range(k)]
            elif t=='JOINT': n.kids.append(pj())
            elif t=='End': nx(); n.kids.append(pj(endsite=True))  # 'Site'
            elif t=='}': break
        return n
    root=None
    while True:
        try: t=nx()
        except StopIteration: break
        if t=='ROOT': root=pj(); break
    return root

# ── recompute offsets ──────────────────────────────────────────────────────────
def assign(node, parent_world, P, stats):
    if node.endsite:
        node.world = [parent_world[i]+node.offset[i] for i in range(3)]
        return
    mn = NAME_MAP.get(node.name)
    if mn is not None and mn in P:                # mapped → MHR rest position
        tw = list(P[mn])
        node.offset = [tw[i]-parent_world[i] for i in range(3)]
        node.world  = tw
        stats[0]+=1
    else:                                         # unmapped → carry template offset
        node.world = [parent_world[i]+node.offset[i] for i in range(3)]
        stats[1]+=1
    for c in node.kids: assign(c, node.world, P, stats)

# ── emit ────────────────────────────────────────────────────────────────────────
def emit(node, depth, lines):
    ind='  '*depth
    if node.endsite:
        lines.append(f'{ind}End Site')
        lines.append(f'{ind}{{')
        lines.append(f'{ind}  OFFSET {node.offset[0]:.6f} {node.offset[1]:.6f} {node.offset[2]:.6f}')
        lines.append(f'{ind}}}')
        return
    kw = 'ROOT' if depth==0 else 'JOINT'
    lines.append(f'{ind}{kw} {node.name}')
    lines.append(f'{ind}{{')
    lines.append(f'{ind}  OFFSET {node.offset[0]:.6f} {node.offset[1]:.6f} {node.offset[2]:.6f}')
    lines.append(f'{ind}  CHANNELS {len(node.channels)} {" ".join(node.channels)}')
    for c in node.kids: emit(c, depth+1, lines)
    lines.append(f'{ind}}}')

def count_channels(node):
    n = 0 if node.endsite else len(node.channels)
    for c in node.kids: n += count_channels(c)
    return n

def main():
    P = mhr_rest_world()
    root = parse(SRC)
    stats=[0,0]
    root.offset=[0,0,0]; root.world=[0,0,0]
    for c in root.kids: assign(c, [0,0,0], P, stats)
    lines=['HIERARCHY']
    emit(root, 0, lines)
    nch = count_channels(root)
    lines += ['MOTION','Frames: 1','Frame Time: 0.04', ' '.join(['0']*nch)]
    open(OUT,'w').write('\n'.join(lines)+'\n')
    print(f"wrote {OUT}")
    print(f"  source template : {SRC}")
    print(f"  mapped joints   : {stats[0]} (set to MHR rest position)")
    print(f"  unmapped joints : {stats[1]} (template offset carried)")
    print(f"  channels/frame  : {nch}")

if __name__ == '__main__':
    main()
