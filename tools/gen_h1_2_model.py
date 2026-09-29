#!/usr/bin/env python3
"""
gen_h1_2_model.py — generate robot/h1_bridge/h1_2_model.h (+ FK golden vectors).

Everything the H1-2 bridge knows about the robot comes from here, derived from
authoritative sources rather than typed by hand:
  • joint limits / velocity limits / origins / axes  ← GMR/assets/unitree_h1_2/h1_2_handless.urdf
  • motor indices + the Unitree reference kp/kd      ← unitree_sdk2 example/h1/high_level/
                                                        h1_2_arm_sdk_dds_example.cpp
  • self-collision boxes (torso+head, pelvis, thighs, shins) ← the MJCF visual-mesh AABBs
  • FK golden vectors for the unit test               ← MuJoCo forward kinematics

Run with the GMR venv (needs mujoco):
    GMR/venv/bin/python tools/gen_h1_2_model.py [--sdk ../unitree_sdk2]
"""
import argparse, re, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--sdk", default=str(REPO.parent / "unitree_sdk2"))
ap.add_argument("--out-dir", default=str(REPO / "robot" / "h1_bridge"))
ap.add_argument("--n-golden", type=int, default=64)
a = ap.parse_args()

URDF = REPO / "GMR/assets/unitree_h1_2/h1_2_handless.urdf"
MJCF = REPO / "GMR/assets/unitree_h1_2/h1_2_handless.xml"
EXAMPLE = Path(a.sdk) / "example/h1/high_level/h1_2_arm_sdk_dds_example.cpp"

# Controlled joints, in the bridge's canonical order (== the SDK example's arm_joints order).
SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINTS = [f"{s}_{j}_joint" for s in SIDES for j in ARM] + ["torso_joint"]
ENUM = [f"k{s.capitalize()}{''.join(w.capitalize() for w in j.split('_'))}" for s in SIDES for j in ARM] + ["kWaistYaw"]

# ── SDK example: motor indices, weight index, kp/kd ───────────────────────────
src = EXAMPLE.read_text()
enum = {m.group(1): int(m.group(2)) for m in re.finditer(r"\b(k\w+)\s*=\s*(\d+)\s*,", src)}
motor = [enum[e] for e in ENUM]
weight_idx = enum["kNotUsedJoint"]
def arr(name):
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", src)
    return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
kp, kd = arr("kp_array"), arr("kd_array")
assert len(kp) == len(kd) == len(JOINTS) == 15, (len(kp), len(kd))

# ── URDF: limits + kinematic chain ────────────────────────────────────────────
root = ET.parse(URDF).getroot()
uj = {j.get("name"): j for j in root.findall("joint")}
def f3(s): return [float(x) for x in s.split()]
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])
spec = []
for n in JOINTS:
    j = uj[n]; lim = j.find("limit"); o = j.find("origin")
    spec.append(dict(name=n, lo=float(lim.get("lower")), hi=float(lim.get("upper")),
                     vel=float(lim.get("velocity")), eff=float(lim.get("effort")),
                     xyz=f3(o.get("xyz")), R=rpy_to_R(*f3(o.get("rpy") or "0 0 0")),
                     axis=f3(j.find("axis").get("xyz"))))

# ── MJCF: collision boxes (pelvis frame, legs at the standing zero pose) ──────
import mujoco
m = mujoco.MjModel.from_xml_path(str(MJCF)); d = mujoco.MjData(m)
d.qpos[:] = 0; d.qpos[3] = 1.0; d.qpos[2] = 0.0       # pelvis at the origin, identity
mujoco.mj_forward(m, d)
def mesh_box(body, frame_body):
    """Largest visual-mesh AABB of `body`, expressed as an axis-aligned box in
    `frame_body`'s frame (which is the pelvis or the torso)."""
    best = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != m.body(body).id or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        if best is None or np.prod(h) > np.prod(best[1]): best = (g, h, c)
    g, h, c = best
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    Rg = d.geom_xmat[g].reshape(3, 3); pg = d.geom_xpos[g]
    fb = m.body(frame_body).id
    Rf = d.xmat[fb].reshape(3, 3); pf = d.xpos[fb]
    pts = (Rf.T @ ((corners @ Rg.T + pg) - pf).T).T
    return pts.min(0), pts.max(0)
boxes = [("torso_head", "torso_link", "torso_link"), ("pelvis", "pelvis", "pelvis"),
         ("left_thigh", "left_hip_roll_link", "pelvis"), ("right_thigh", "right_hip_roll_link", "pelvis"),
         ("left_shin", "left_knee_link", "pelvis"), ("right_shin", "right_knee_link", "pelvis")]
box_vals = [(n, fr, *mesh_box(b, fr)) for n, b, fr in boxes]

# ── golden FK vectors: q (15) -> elbow + wrist_yaw origins, pelvis frame ──────
rng = np.random.default_rng(0)
qadr = [m.jnt_qposadr[m.joint(n).id] for n in JOINTS]
golden = []
for _ in range(a.n_golden):
    q = np.array([rng.uniform(s["lo"], s["hi"]) for s in spec])
    d.qpos[:] = 0; d.qpos[3] = 1.0
    for i, adr in enumerate(qadr): d.qpos[adr] = q[i]
    mujoco.mj_forward(m, d)
    pts = [d.xpos[m.body(f"{s}_{b}").id].copy() for s in SIDES for b in ("elbow_link", "wrist_yaw_link")]
    golden.append((q, pts))

# ── emit ───────────────────────────────────────────────────────────────────────
def fl(x):
    t = f"{float(x):.9g}"
    if not any(c in t for c in ".en"): t += ".0"   # C++ needs 9.0f, not 9f
    return t + "f"
def v3(v): return "{" + ", ".join(fl(x) for x in v) + "}"
def m33(M): return "{" + ", ".join(v3(r) for r in M) + "}"
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
H = [f"// GENERATED by tools/gen_h1_2_model.py — do not edit by hand.",
     f"// Sources: {URDF.relative_to(REPO)}, {MJCF.relative_to(REPO)},",
     f"//          unitree_sdk2/{EXAMPLE.relative_to(Path(a.sdk))}",
     "#pragma once", "#include <array>", "", "namespace h1b {", "",
     f"constexpr int kNumJoints = {len(JOINTS)};      // 7 left arm, 7 right arm, waist",
     f"constexpr int kArmSdkWeightIndex = {weight_idx}; // motor_cmd[k].q carries the arm_sdk blend weight",
     f"constexpr int kLowStateMotors = 35;", "",
     "struct JointSpec {",
     "    const char* name;      // URDF / GMR joint name",
     "    int   motor;           // index into LowCmd_/LowState_ motor arrays",
     "    float lo, hi;          // URDF position limits (rad)",
     "    float vel;             // URDF velocity limit (rad/s)",
     "    float kp, kd;          // Unitree reference gains (h1_2_arm_sdk_dds_example) = hard ceiling",
     "    float origin[3];       // joint origin in the parent link (m)",
     "    float R[3][3];         // joint origin rotation in the parent link",
     "    float axis[3];         // rotation axis in the child frame",
     "};", "",
     "inline constexpr std::array<JointSpec, kNumJoints> kJoints = {{"]
for s, mi, p, dd in zip(spec, motor, kp, kd):
    H.append(f"    {{\"{s['name']}\", {mi}, {fl(s['lo'])}, {fl(s['hi'])}, {fl(s['vel'])}, {fl(p)}, {fl(dd)},")
    H.append(f"     {v3(s['xyz'])}, {m33(s['R'])}, {v3(s['axis'])}}},")
H += ["}};", "",
      "enum JointIdx { kL0 = 0, kR0 = 7, kWaist = 14 };", "",
      "// Self-collision boxes from the visual-mesh AABBs (legs assumed at the standing zero",
      "// pose). frame: 0 = pelvis, 1 = torso_link (rotates with the waist).",
      "struct CollisionBox { const char* name; int frame; float lo[3]; float hi[3]; };",
      f"inline constexpr std::array<CollisionBox, {len(box_vals)}> kBoxes = {{{{"]
for n, fr, lo, hi in box_vals:
    H.append(f"    {{\"{n}\", {0 if fr == 'pelvis' else 1}, {v3(lo)}, {v3(hi)}}},")
H += ["}};", "", "}  // namespace h1b", ""]
(out / "h1_2_model.h").write_text("\n".join(H))

G = ["// GENERATED by tools/gen_h1_2_model.py — MuJoCo FK reference for test_bridge.cpp.",
     "#pragma once", "#include <array>", "namespace h1b_golden {",
     "struct Case { float q[15]; float p[4][3]; };  // p: L elbow, L wrist_yaw, R elbow, R wrist_yaw",
     f"inline constexpr std::array<Case, {len(golden)}> kCases = {{{{"]
for q, pts in golden:
    G.append("    {{" + ", ".join(fl(x) for x in q) + "}, {" + ", ".join(v3(p) for p in pts) + "}},")
G += ["}};", "}", ""]
(out / "h1_2_fk_golden.h").write_text("\n".join(G))

print(f"wrote {out/'h1_2_model.h'} and {out/'h1_2_fk_golden.h'}")
print(f"motors={motor}  weight_idx={weight_idx}")
for n, fr, lo, hi in box_vals:
    print(f"  box {n:12s} [{fr}]  lo={np.round(lo,3)}  hi={np.round(hi,3)}")
