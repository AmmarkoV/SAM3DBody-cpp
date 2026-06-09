#!/usr/bin/env python3
"""
gen_lafan_bvh.py — generate lafan_mhr.bvh from the MHR rest skeleton.

Sibling of gen_mhr_bvh.py, but for the LAFAN1 / GMR template.  Takes the
hand-authored lafan.bvh (LAFAN1 joint names + hierarchy + uniform ZXY channels)
and rewrites every mapped joint's OFFSET so its rest position equals its
corresponding MHR joint's rest position (from onnx/body_model.lbs).  The result
keeps the LAFAN1 names (so GMR's bvh_lafan1 reader + our NAME_MAP still work)
and the ZXY channel order, but its REST POSE matches MHR exactly.

Why: the original lafan.bvh is an *anatomical* T-pose whose rest bone directions
differ from MHR by large angles (thigh ~62°, forearm ~48°, hand ~45°).  The BVH
writer's rest retarget (q_bone_align) only corrects bone *swing*, not twist, and
shares one alignment across *branch* joints — so those big rest-pose mismatches
come out as the twisted torso / splayed legs / wrong arm bends we saw after GMR.
When the template rest pose == the MHR rest pose, q_bone_align becomes identity,
the writer faithfully reproduces MHR world joint positions, and GMR's
position-based config then retargets clean targets.  (Same trick gen_mhr_bvh.py
uses for body_mhr.bvh — that's exactly why body_mhr works and lafan didn't.)

Usage:
    python3 tools/gen_lafan_bvh.py [lafan.bvh] [onnx/body_model.lbs] [out=lafan_mhr.bvh]

Mapped joints land exactly at MHR rest positions; unmapped joints (LeftToe /
RightToe — left flat by design) and End Sites keep their template offsets and are
carried along.  Root (Hips) stays at the origin.
"""

import sys, os, struct, re, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'lafan.bvh')
LBS  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'onnx', 'body_model.lbs')
OUT  = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'lafan_mhr.bvh')

# ── NAME_MAP (LAFAN1 BVH name → MHR name), mirrors src/bvh_writer.cpp (lafan rows)
NAME_MAP = {
    'Hips':'root',
    'Spine':'c_spine1','Spine1':'c_spine2','Spine2':'c_spine3',
    'Neck':'c_neck','Head':'c_head',
    'LeftShoulder':'l_clavicle','LeftArm':'l_uparm','LeftForeArm':'l_lowarm','LeftHand':'l_wrist',
    'RightShoulder':'r_clavicle','RightArm':'r_uparm','RightForeArm':'r_lowarm','RightHand':'r_wrist',
    'LeftUpLeg':'l_upleg','LeftLeg':'l_lowleg','LeftFoot':'l_foot',
    'RightUpLeg':'r_upleg','RightLeg':'r_lowleg','RightFoot':'r_foot',
    'LeftHandThumb1':'l_thumb1','LeftHandThumb2':'l_thumb2','LeftHandThumb3':'l_thumb3',
    'LeftHandIndex1':'l_index1','LeftHandIndex2':'l_index2','LeftHandIndex3':'l_index3',
    'LeftHandMiddle1':'l_middle1','LeftHandMiddle2':'l_middle2','LeftHandMiddle3':'l_middle3',
    'LeftHandRing1':'l_ring1','LeftHandRing2':'l_ring2','LeftHandRing3':'l_ring3',
    'LeftHandPinky1':'l_pinky1','LeftHandPinky2':'l_pinky2','LeftHandPinky3':'l_pinky3',
    'RightHandThumb1':'r_thumb1','RightHandThumb2':'r_thumb2','RightHandThumb3':'r_thumb3',
    'RightHandIndex1':'r_index1','RightHandIndex2':'r_index2','RightHandIndex3':'r_index3',
    'RightHandMiddle1':'r_middle1','RightHandMiddle2':'r_middle2','RightHandMiddle3':'r_middle3',
    'RightHandRing1':'r_ring1','RightHandRing2':'r_ring2','RightHandRing3':'r_ring3',
    'RightHandPinky1':'r_pinky1','RightHandPinky2':'r_pinky2','RightHandPinky3':'r_pinky3',
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
