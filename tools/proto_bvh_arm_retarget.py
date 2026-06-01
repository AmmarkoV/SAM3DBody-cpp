#!/usr/bin/env python3
"""
proto_bvh_arm_retarget.py — offline prototype for the BVH arm-retarget fix.

Compares, per mapped bone, the MHR ground-truth world bone DIRECTION against the
BVH bone direction produced by:
  (A) the current retarget   : delta-from-rest, optionally conjugated by q_align
  (B) the corrected retarget : bake q_align onto the PARENT's world rotation so the
                               BVH bone points where the MHR bone actually points

Ground truth = MHR forward kinematics from the LBS data + the per-frame model_params
dump (/tmp/mhr_lbs_dump.bin, written on the first frame of any --lbs run).  This is
the same pose that drives the deformed mesh, so it is the correct target.

Bone directions are frame/offset independent, so cam_t and bone lengths drop out —
we only need the LBS data and the BVH template hierarchy.

Run:
    python3 tools/proto_bvh_arm_retarget.py [body.bvh] [/tmp/mhr_lbs_dump.bin]
"""

import sys, struct, math

LBS_PATH  = sys.argv[2] if len(sys.argv) > 2 else 'onnx/body_model.lbs'
DUMP_PATH = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mhr_lbs_dump.bin'
BVH_PATH  = sys.argv[1] if len(sys.argv) > 1 else 'body.bvh'
LBS_PATH  = 'onnx/body_model.lbs'   # fixed; dump path stays /tmp

# ── quaternion helpers (match bvh_writer.cpp exactly) ─────────────────────────
# qmul(a,b) = Hamilton a⊗b ; rotating a vector applies b first then a (a = parent)

def qmul(a, b):
    return [
        b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2],
        b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2],
        b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2],
        b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2],
    ]

def qconj(a): return [-a[0], -a[1], -a[2], a[3]]

def qrot(q, v):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0*(qy*vz - qz*vy)
    ty = 2.0*(qz*vx - qx*vz)
    tz = 2.0*(qx*vy - qy*vx)
    return [vx + qw*tx + (qy*tz - qz*ty),
            vy + qw*ty + (qz*tx - qx*tz),
            vz + qw*tz + (qx*ty - qy*tx)]

def euler_mhr_to_quat(ex, ey, ez):
    hx, hy, hz = ex*0.5, ey*0.5, ez*0.5
    qx = [math.sin(hx), 0, 0, math.cos(hx)]
    qy = [0, math.sin(hy), 0, math.cos(hy)]
    qz = [0, 0, math.sin(hz), math.cos(hz)]
    return qmul(qmul(qz, qy), qx)

def shortest_arc_quat(frm, to):
    fn = math.sqrt(sum(c*c for c in frm))
    tn = math.sqrt(sum(c*c for c in to))
    if fn < 1e-9 or tn < 1e-9: return [0,0,0,1]
    u = [c/fn for c in frm]; v = [c/tn for c in to]
    d = sum(u[i]*v[i] for i in range(3))
    if d >  0.999999: return [0,0,0,1]
    if d < -0.999999:
        ax = [1,0,0]
        if abs(u[0]) >= 0.9: ax = [0,1,0]
        c = [u[1]*ax[2]-u[2]*ax[1], u[2]*ax[0]-u[0]*ax[2], u[0]*ax[1]-u[1]*ax[0]]
        cn = math.sqrt(sum(x*x for x in c)) + 1e-12
        return [c[0]/cn, c[1]/cn, c[2]/cn, 0]
    c = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    q = [c[0], c[1], c[2], 1.0+d]
    n = math.sqrt(sum(x*x for x in q))
    return [x/n for x in q]

def vsub(a, b): return [a[i]-b[i] for i in range(3)]
def vnorm(a):
    n = math.sqrt(sum(x*x for x in a)) + 1e-12
    return [x/n for x in a]
def angle_between(a, b):
    a, b = vnorm(a), vnorm(b)
    d = max(-1.0, min(1.0, sum(a[i]*b[i] for i in range(3))))
    return math.degrees(math.acos(d))


# ── load LBS data ─────────────────────────────────────────────────────────────

def load_lbs(path):
    with open(path, 'rb') as f:
        data = f.read()
    hdr = struct.unpack_from('8I', data, 0)
    nj, n_skin, n_verts, n_shape, n_face, pt_cols = hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]
    pt_rows = nj * 7
    off = 32
    PT = struct.unpack_from(f'{pt_rows*pt_cols}f', data, off); off += pt_rows*pt_cols*4
    joint_offsets = struct.unpack_from(f'{nj*3}f', data, off); off += nj*3*4
    prerot = struct.unpack_from(f'{nj*4}f', data, off); off += nj*4*4
    parents = struct.unpack_from(f'{nj}i', data, off); off += nj*4
    return dict(nj=nj, pt_cols=pt_cols, pt_rows=pt_rows, PT=PT,
                joint_offsets=joint_offsets, prerot=prerot, parents=parents)

def load_dump(path):
    with open(path, 'rb') as f:
        data = f.read()
    nj, npc = struct.unpack_from('2i', data, 0)
    model_params = list(struct.unpack_from('204f', data, 8))
    return model_params


# ── MHR forward kinematics (matches compute_per_frame_mhr_state) ──────────────

def mhr_fk(lbs, model_params, rest=False):
    nj, pt_cols, PT = lbs['nj'], lbs['pt_cols'], lbs['PT']
    prerot, parents, joint_offsets = lbs['prerot'], lbs['parents'], lbs['joint_offsets']
    take = min(pt_cols, 204)

    # joint_params = PT @ model_params  (only rotations used here, but compute t too)
    jp = [0.0]*(nj*7)
    for row in range(nj*7):
        base = row*pt_cols
        jp[row] = sum(PT[base+k]*model_params[k] for k in range(take))

    g_q = [[0,0,0,1] for _ in range(nj)]
    g_t = [[0,0,0] for _ in range(nj)]
    for j in range(nj):
        pre = list(prerot[j*4:j*4+4])
        if rest:
            q_local = pre[:]            # rest = prerotation only (no pose euler)
            jt = [0.0, 0.0, 0.0]
        else:
            jpj = jp[j*7:j*7+7]
            q_euler = euler_mhr_to_quat(jpj[3], jpj[4], jpj[5])
            q_local = qmul(pre, q_euler)
            jt = [jpj[0], jpj[1], jpj[2]]
        off = joint_offsets[j*3:j*3+3]
        local_off = [off[0]+jt[0], off[1]+jt[1], off[2]+jt[2]]
        p = parents[j]
        if p < 0 or p >= nj:
            g_q[j] = q_local
            g_t[j] = local_off
        else:
            g_q[j] = qmul(g_q[p], q_local)
            rt = qrot(g_q[p], local_off)
            g_t[j] = [g_t[p][i]+rt[i] for i in range(3)]
    return g_q, g_t


# ── BVH template parser ───────────────────────────────────────────────────────

class Joint:
    def __init__(s, name, parent): s.name=name; s.parent=parent; s.children=[]; s.offset=[0,0,0]; s.channels=[]

def parse_bvh(path):
    toks = open(path).read().split()
    it = iter(toks); nxt = lambda: next(it)
    joints = []
    def pj(parent):
        name = nxt(); j = Joint(name, parent); joints.append(j)
        assert nxt() == '{'
        while True:
            t = nxt()
            if t == 'OFFSET': j.offset = [float(nxt()), float(nxt()), float(nxt())]
            elif t == 'CHANNELS':
                n = int(nxt()); j.channels = [nxt() for _ in range(n)]
            elif t == 'JOINT': j.children.append(pj(j))
            elif t == 'End':
                nxt(); nxt()  # 'Site' '{'
                while nxt() != '}': pass
            elif t == '}': break
        return j
    root = None
    while True:
        try: t = nxt()
        except StopIteration: break
        if t == 'ROOT': root = pj(None); break
    return root, joints


# ── name maps ─────────────────────────────────────────────────────────────────

NAME_MAP = {
    'hip':'root','abdomen':'c_spine1','chest':'c_spine3','neck':'c_neck','head':'c_head',
    'lCollar':'l_clavicle','lShldr':'l_uparm','lForeArm':'l_lowarm','lHand':'l_wrist',
    'rCollar':'r_clavicle','rShldr':'r_uparm','rForeArm':'r_lowarm','rHand':'r_wrist',
    'lThigh':'l_upleg','lShin':'l_lowleg','lFoot':'l_foot',
    'rThigh':'r_upleg','rShin':'r_lowleg','rFoot':'r_foot',
}

def mhr_names():
    import re
    txt = open('src/mhr_joint_table.h').read()
    block = txt.split('NAMES[N_JOINTS] = {',1)[1].split('};',1)[0]
    return [m for m in re.findall(r'"([^"]+)"', block)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    lbs = load_lbs(LBS_PATH)
    model_params = load_dump(DUMP_PATH)
    mhr_name_to_idx = {n:i for i,n in enumerate(mhr_names())}

    g_q, g_t = mhr_fk(lbs, model_params, rest=False)
    gqr, gtr = mhr_fk(lbs, model_params, rest=True)

    root, joints = parse_bvh(BVH_PATH)
    by_name = {j.name:j for j in joints}

    # Build slots: each BVH joint that maps to an MHR joint, plus nearest mapped ancestor.
    def nearest_mapped_ancestor(j):
        p = j.parent
        while p is not None:
            if p.name in NAME_MAP: return p
            p = p.parent
        return None

    # BVH rest world positions (offsets accumulated, identity rotations)
    bvh_rest = {}
    def accum(j, base):
        pos = [base[i]+j.offset[i] for i in range(3)]
        bvh_rest[j.name] = pos
        for c in j.children: accum(c, pos)
    accum(root, [0,0,0])

    # Per-slot q_align (shortest arc: MHR rest bone dir → BVH rest bone dir)
    slots = []  # (bvh_name, mhr_idx, anc_bvh_name, anc_mhr_idx, q_align)
    for jn, mn in NAME_MAP.items():
        if jn not in by_name or mn not in mhr_name_to_idx: continue
        m = mhr_name_to_idx[mn]
        anc = nearest_mapped_ancestor(by_name[jn])
        if anc is None: continue
        am = mhr_name_to_idx[NAME_MAP[anc.name]]
        md = vsub(gtr[m], gtr[am])                      # MHR rest bone dir (world)
        bd = vsub(bvh_rest[jn], bvh_rest[anc.name])     # BVH rest bone dir (world)
        qa = shortest_arc_quat(md, bd)
        slots.append((jn, m, anc.name, am, qa))

    # delta from MHR rest, global frame
    def delta(m): return qmul(g_q[m], qconj(gqr[m]))

    # ── Method A (current): q_local_bvh = inv(delta_par)·delta_self, opt conj q_align
    # ── Method B (corrected): target BVH global = delta(parent)·q_align⁻¹ controls the
    #    bone; we realise it as a per-joint global orientation then take bone dirs.
    #
    # For a direction check we only need each bone's WORLD direction, which is set by
    # the PARENT joint's world rotation applied to the BVH rest bone dir.
    #   A: parent world rot = product down the chain of inv(delta_par)·delta_self … but
    #      equivalently the accumulated BVH global = delta(parent_mhr) (telescopes).
    #      With conjugation, each local gets q_align·R·q_align⁻¹.
    #   B: we want BVH bone dir = MHR bone dir exactly → parent global = delta·q_align⁻¹.
    #
    # We compute bone dirs directly:
    #   mhr_dir   = g_t[m] - g_t[am]                      (ground truth, world)
    #   bvhA_dir  = R_globalA(anc) · bvh_rest_dir
    #   bvhB_dir  = R_globalB(anc) · bvh_rest_dir
    # where R_globalA(anc) = delta(anc_mhr) conjugated-chain, R_globalB(anc)=delta(anc_mhr)·qa⁻¹

    # ── Realizable assignment: ONE world rotation per mapped joint ────────────
    # A joint controls all its mapped child bones with a single rotation.  For a
    # single-child joint we can aim that child perfectly (R = delta·qa⁻¹).  For a
    # multi-child joint (e.g. chest → neck + both clavicles) one rotation cannot
    # satisfy all; we pick the longest child bone as primary and measure the
    # honest residual on the others.
    children_of = {}   # parent_bvh_name → list of (child_jn, child_m, qa, rest_len)
    for jn, m, ancn, am, qa in slots:
        rest_len = math.sqrt(sum(c*c for c in vsub(bvh_rest[jn], bvh_rest[ancn])))
        children_of.setdefault(ancn, []).append((jn, m, qa, rest_len))

    # Targeted, low-risk plan: a joint bakes its child's q_align ONLY when it has
    # exactly one mapped child (a single-child chain — the arm below the collar,
    # and the lower leg).  Branch joints (chest→neck+2 clavicles, hip→2 thighs+
    # spine) stay on the current method, since the overlay already shows their
    # downstream joint positions are correct.  This is conflict-free and cannot
    # regress the working torso/legs.
    C = {}  # parent_bvh_name → qa to bake (identity = keep current method)
    for ancn, kids in children_of.items():
        C[ancn] = kids[0][2] if len(kids) == 1 else [0,0,0,1]

    print(f"{'bone (BVH)':12s} {'mhr_joint':12s} {'A err°':>7s} {'B err°':>7s}   verdict")
    print('-'*62)
    sumA = sumB = 0.0; nA = 0
    for jn, m, ancn, am, qa in slots:
        mhr_dir = vsub(g_t[m], g_t[am])
        bvh_rest_dir = vsub(bvh_rest[jn], bvh_rest[ancn])

        # Method A (current): parent global telescopes to delta(am); local conj by
        # q_align of THIS joint, but the bone dir is governed by the parent global.
        RA = delta(am)
        bvhA_dir = qrot(RA, bvh_rest_dir)

        # Method B (realizable): parent global = delta(am) · C[parent]⁻¹, where
        # C[parent] is the parent's single chosen alignment (its primary child).
        RB = qmul(delta(am), qconj(C[ancn]))
        bvhB_dir = qrot(RB, bvh_rest_dir)

        eA = angle_between(mhr_dir, bvhA_dir)
        eB = angle_between(mhr_dir, bvhB_dir)
        sumA += eA; sumB += eB; nA += 1
        nkid = len(children_of[ancn])
        tag = '' if nkid == 1 else f'(parent has {nkid} kids)'
        verdict = ('B fixes' if eB + 2 < eA else
                   ('~same' if abs(eA-eB) <= 2 else 'B WORSE'))
        print(f"{jn:12s} {mhr_names()[m]:12s} {eA:7.1f} {eB:7.1f}   {verdict:8s}{tag}")

    print('-'*62)
    print(f"{'MEAN':12s} {'':12s} {sumA/nA:7.1f} {sumB/nA:7.1f}")
    print()
    print("err° = angle between MHR ground-truth bone direction and the BVH bone.")
    print("A = current export.  B = targeted fix: bake child q_align only on single-child")
    print("chain joints (arm below collar, lower leg).  Branch joints (chest/hip) unchanged,")
    print("so torso/leg positions cannot regress.  Collar bone is not addressed (= same as A).")


if __name__ == '__main__':
    main()
