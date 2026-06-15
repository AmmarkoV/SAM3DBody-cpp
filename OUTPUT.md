# OUTPUT — 2D / 3D points and joint labels

This document is a reference for every set of **points** (2D and 3D) and their
**labels** that `SAM3DBody-cpp` produces for each detected person. It pulls the
definitions together from:

- `src/fast_sam_3dbody.h` / `fast_sam_3dbody_capi.h` — the `MHRResult` / `FsbResult` struct
- `fast_sam_3dbody_dump_csv.py` — the canonical **MHR-70** keypoint name list
- `fast_sam_3dbody_frontend.py` — the **COCO-17** keypoint name list
- `src/mhr_joint_table.h` — the **127** MHR LBS skeleton joint names + parents
- `render/fast_sam_3dbody_render.cpp` — the `.obj` / `.joints` export formats

---

## 1. Per-person result struct

One of these is produced per detected person per frame (`MHRResult` in C++,
`FsbResult` in the C/ctypes API — identical layout):

| Field | Shape | Space / units | Description |
|-------|-------|---------------|-------------|
| `bbox` | [4] | image pixels | `x1 y1 x2 y2` in the original image |
| `focal_length` | scalar | pixels | Estimated focal length |
| `pred_cam_t` | [3] | — | Raw camera head output `[s, tx, ty]` |
| `global_rot` | [3] | radians | Global orientation, Euler **ZYX** |
| `body_pose` | [133] | radians | Body joint Euler angles |
| `shape` | [45] | — | SMPL-like identity blend-shape betas |
| `scale` | [28] | — | Scale PCA components |
| `hand_pose` | [108] | radians | Hands: left [54] + right [54] |
| `face_params` | [72] | — | Facial expression parameters |
| `yolo_kps` | [51] | image pixels | **COCO-17** × `[x, y, conf]` (see §3) |
| `has_yolo_kps` | int | — | 1 if `yolo_kps` valid |
| `kps_3d` | [210] | **metres** | **MHR-70** joints × `[x, y, z]` (see §2) |
| `kps_2d` | [140] | image pixels | **MHR-70** joints × `[x, y]` projected (see §2) |
| `has_kps` | int | — | 1 if `kps_2d`/`kps_3d` valid (native LBS ran) |
| `pred_vertices` | [55317] | metres | 18439 mesh verts × 3 (empty if `--skip-body`) |

`kps_2d`, `kps_3d` and `pred_vertices` are only populated when the native C LBS
body model runs (i.e. **not** `--skip-body`). `yolo_kps` is always available.

There are therefore **three** independent point sets per person:

1. **COCO-17** — `yolo_kps`, straight from the YOLO pose detector (2D + conf).
2. **MHR-70** — `kps_2d` (2D) and `kps_3d` (3D), regressed from the body model.
3. **MHR-127** — the full LBS skeleton joint centres, written by the renderer to
   `.joints` files (3D only).

---

## 2. MHR-70 keypoints (`kps_3d` / `kps_2d`)

70 joints. `kps_3d` is `70 × [x, y, z]` in metres (camera space); `kps_2d` is the
same 70 joints projected to `70 × [x, y]` image pixels. Order matches
`MHR70_NAMES` in `fast_sam_3dbody_dump_csv.py` (from
`sam_3d_body/metadata/mhr70.py original_keypoint_info`).

**Layout:** `0–20` body/foot · `21–41` right hand · `42–62` left hand · `63–69` extra.

| Idx | Label | Idx | Label | Idx | Label |
|----:|-------|----:|-------|----:|-------|
| 0 | nose | 24 | right_thumb_third_joint | 48 | left_index_second_joint |
| 1 | left_eye | 25 | right_index_tip | 49 | left_index_third_joint |
| 2 | right_eye | 26 | right_index_first_joint | 50 | left_middle_tip |
| 3 | left_ear | 27 | right_index_second_joint | 51 | left_middle_first_joint |
| 4 | right_ear | 28 | right_index_third_joint | 52 | left_middle_second_joint |
| 5 | left_shoulder | 29 | right_middle_tip | 53 | left_middle_third_joint |
| 6 | right_shoulder | 30 | right_middle_first_joint | 54 | left_ring_tip |
| 7 | left_elbow | 31 | right_middle_second_joint | 55 | left_ring_first_joint |
| 8 | right_elbow | 32 | right_middle_third_joint | 56 | left_ring_second_joint |
| 9 | left_hip | 33 | right_ring_tip | 57 | left_ring_third_joint |
| 10 | right_hip | 34 | right_ring_first_joint | 58 | left_pinky_tip |
| 11 | left_knee | 35 | right_ring_second_joint | 59 | left_pinky_first_joint |
| 12 | right_knee | 36 | right_ring_third_joint | 60 | left_pinky_second_joint |
| 13 | left_ankle | 37 | right_pinky_tip | 61 | left_pinky_third_joint |
| 14 | right_ankle | 38 | right_pinky_first_joint | 62 | left_wrist |
| 15 | left_big_toe_tip | 39 | right_pinky_second_joint | 63 | left_olecranon |
| 16 | left_small_toe_tip | 40 | right_pinky_third_joint | 64 | right_olecranon |
| 17 | left_heel | 41 | right_wrist | 65 | left_cubital_fossa |
| 18 | right_big_toe_tip | 42 | left_thumb_tip | 66 | right_cubital_fossa |
| 19 | right_small_toe_tip | 43 | left_thumb_first_joint | 67 | left_acromion |
| 20 | right_heel | 44 | left_thumb_second_joint | 68 | right_acromion |
| 21 | right_thumb_tip | 45 | left_thumb_third_joint | 69 | neck |
| 22 | right_thumb_first_joint | 46 | left_index_tip | | |
| 23 | right_thumb_second_joint | 47 | left_index_first_joint | | |

> Note: MHR-70 **wrists** are at indices **62 (left)** and **41 (right)** — at the
> end of each hand block, not in the body block. Index 9/10 are **hips**, not wrists.
> The CSV exporter (`fast_sam_3dbody_dump_csv.py`) writes these 70 names ×
> `_3DX,_3DY,_3DZ` columns, one frame per row, `0,0,0` when a joint is absent.

### COCO-17 → MHR-70 index map

The lightweight frontend reconciles the two sets with
(`_COCO_TO_MHR70` in `fast_sam_3dbody_frontend.py`):

```
COCO idx:   0  1  2  3  4  5  6  7  8   9  10  11 12 13 14 15 16
MHR70 idx:  0  1  2  3  4  5  6  7  8  62  41   9 10 11 12 13 14
```

(COCO wrists 9/10 map to MHR-70 62/41; COCO hips 11/12 map to MHR-70 9/10.)

---

## 3. COCO-17 keypoints (`yolo_kps`)

51 floats = 17 joints × `[x, y, confidence]`, in original-image pixels. Standard
COCO person keypoint order (labels/colours from `fast_sam_3dbody_frontend.py`):

| Idx | Label | Idx | Label |
|----:|-------|----:|-------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

**Skeleton edges** (`COCO_EDGES`):
`(0,1)(0,2)(1,3)(2,4)(0,5)(0,6)(5,6)(5,7)(7,9)(6,8)(8,10)(5,11)(6,12)(11,12)(11,13)(13,15)(12,14)(14,16)`

---

## 4. MHR-127 LBS skeleton joints (`.joints` export)

The full body-model skeleton has **127** named joints (`src/mhr_joint_table.h`,
generated by `tools/build_joint_table.py`). When the OpenGL renderer dumps a
person it writes a `<name>_p<id>_<frame>.joints` file with one
`idx x y z` line per joint (127 lines), in the same world space as the `.obj`
mesh (see §5). These are the LBS rotation-centre joints — the ones BVH export
maps against — distinct from the 70 surface landmarks of MHR-70.

| Idx | Name | Parent | Idx | Name | Parent |
|----:|------|-------:|----:|------|-------:|
| 0 | body_world | -1 | 64 | r_lowarm_twist1_proc | 40 |
| 1 | root | 0 | 65 | r_lowarm_twist2_proc | 40 |
| 2 | l_upleg | 1 | 66 | r_lowarm_twist3_proc | 40 |
| 3 | l_lowleg | 2 | 67 | r_lowarm_twist4_proc | 40 |
| 4 | l_foot | 3 | 68 | r_uparm_twist0_proc | 39 |
| 5 | l_talocrural | 4 | 69 | r_uparm_twist1_proc | 39 |
| 6 | l_subtalar | 5 | 70 | r_uparm_twist2_proc | 39 |
| 7 | l_transversetarsal | 6 | 71 | r_uparm_twist3_proc | 39 |
| 8 | l_ball | 7 | 72 | r_uparm_twist4_proc | 39 |
| 9 | l_lowleg_twist1_proc | 3 | 73 | l_clavicle | 37 |
| 10 | l_lowleg_twist2_proc | 3 | 74 | l_uparm | 73 |
| 11 | l_lowleg_twist3_proc | 3 | 75 | l_lowarm | 74 |
| 12 | l_lowleg_twist4_proc | 3 | 76 | l_wrist_twist | 75 |
| 13 | l_upleg_twist0_proc | 2 | 77 | l_wrist | 76 |
| 14 | l_upleg_twist1_proc | 2 | 78 | l_pinky0 | 77 |
| 15 | l_upleg_twist2_proc | 2 | 79 | l_pinky1 | 78 |
| 16 | l_upleg_twist3_proc | 2 | 80 | l_pinky2 | 79 |
| 17 | l_upleg_twist4_proc | 2 | 81 | l_pinky3 | 80 |
| 18 | r_upleg | 1 | 82 | l_pinky_null | 81 |
| 19 | r_lowleg | 18 | 83 | l_ring1 | 78 |
| 20 | r_foot | 19 | 84 | l_ring2 | 83 |
| 21 | r_talocrural | 20 | 85 | l_ring3 | 84 |
| 22 | r_subtalar | 21 | 86 | l_ring_null | 85 |
| 23 | r_transversetarsal | 22 | 87 | l_middle1 | 78 |
| 24 | r_ball | 23 | 88 | l_middle2 | 87 |
| 25 | r_lowleg_twist1_proc | 19 | 89 | l_middle3 | 88 |
| 26 | r_lowleg_twist2_proc | 19 | 90 | l_middle_null | 89 |
| 27 | r_lowleg_twist3_proc | 19 | 91 | l_index1 | 78 |
| 28 | r_lowleg_twist4_proc | 19 | 92 | l_index2 | 91 |
| 29 | r_upleg_twist0_proc | 18 | 93 | l_index3 | 92 |
| 30 | r_upleg_twist1_proc | 18 | 94 | l_index_null | 93 |
| 31 | r_upleg_twist2_proc | 18 | 95 | l_thumb0 | 78 |
| 32 | r_upleg_twist3_proc | 18 | 96 | l_thumb1 | 95 |
| 33 | r_upleg_twist4_proc | 18 | 97 | l_thumb2 | 96 |
| 34 | c_spine0 | 1 | 98 | l_thumb3 | 97 |
| 35 | c_spine1 | 34 | 99 | l_thumb_null | 98 |
| 36 | c_spine2 | 35 | 100 | l_lowarm_twist1_proc | 76 |
| 37 | c_spine3 | 36 | 101 | l_lowarm_twist2_proc | 76 |
| 38 | r_clavicle | 37 | 102 | l_lowarm_twist3_proc | 76 |
| 39 | r_uparm | 38 | 103 | l_lowarm_twist4_proc | 76 |
| 40 | r_lowarm | 39 | 104 | l_uparm_twist0_proc | 75 |
| 41 | r_wrist_twist | 40 | 105 | l_uparm_twist1_proc | 75 |
| 42 | r_wrist | 41 | 106 | l_uparm_twist2_proc | 75 |
| 43 | r_pinky0 | 42 | 107 | l_uparm_twist3_proc | 75 |
| 44 | r_pinky1 | 43 | 108 | l_uparm_twist4_proc | 75 |
| 45 | r_pinky2 | 44 | 109 | c_neck | 37 |
| 46 | r_pinky3 | 45 | 110 | c_neck_twist1_proc | 109 |
| 47 | r_pinky_null | 46 | 111 | c_neck_twist0_proc | 109 |
| 48 | r_ring1 | 42 | 112 | c_head | 109 |
| 49 | r_ring2 | 48 | 113 | c_jaw | 112 |
| 50 | r_ring3 | 49 | 114 | c_teeth | 113 |
| 51 | r_ring_null | 50 | 115 | c_jaw_null | 113 |
| 52 | r_middle1 | 42 | 116 | c_tongue0 | 113 |
| 53 | r_middle2 | 52 | 117 | c_tongue1 | 116 |
| 54 | r_middle3 | 53 | 118 | c_tongue2 | 117 |
| 55 | r_middle_null | 54 | 119 | c_tongue3 | 118 |
| 56 | r_index1 | 42 | 120 | c_tongue4 | 119 |
| 57 | r_index2 | 56 | 121 | r_eye | 112 |
| 58 | r_index3 | 57 | 122 | r_eye_null | 121 |
| 59 | r_index_null | 58 | 123 | l_eye | 112 |
| 60 | r_thumb0 | 42 | 124 | l_eye_null | 123 |
| 61 | r_thumb1 | 60 | 125 | c_head_null | 112 |
| 62 | r_thumb2 | 61 | | | |
| 63 | r_thumb3 | 62 | | | |

(`Parent = -1` is the root of the hierarchy. `*_proc` joints are procedural
twist helpers; `*_null` are End-Site tip extensions.)

> The BVH exporter maps a subset (~50) of these to MakeHuman/Mixamo/LAFAN bone
> names via the `NAME_MAP` table in `src/bvh_writer.cpp`; unmapped joints (toes,
> twists, face/tongue, metacarpals) stay at zero rotation.

---

## 5. File output formats

### `.joints` (renderer, `write_mhr_joints`)
One line per LBS joint (127 lines):

```
<idx> <x> <y> <z>
```

World-space coordinates in **cm**, computed as (pelvis-relative, scaled, with
camera translation, Y/Z flipped to BVH world space):

```
x =  (joint.x - pelvis.x) * scale + cam_t.x*scale
y = -(joint.y - pelvis.y) * scale + cam_t.y*scale
z = -(joint.z - pelvis.z) * scale + cam_t.z*scale
```

`idx` indexes the §4 joint table. Example (`buffalo_mesh_p0_00034.joints`):
```
0 1.240793 46.350937 181.054749     # body_world
1 1.240793 138.749634 181.054749    # root
2 9.030171 137.309143 181.970932    # l_upleg
...
```

### `.obj` (renderer, `write_obj_mesh`)
Standard Wavefront OBJ — 18439 `v x y z` vertex lines + faces, same world space
and scale as the `.joints` file above.

### `.csv` (`-o`/`--out`, or `fast_sam_3dbody_dump_csv.py`)
Header is the 70 MHR-70 names each expanded to three columns
(`<name>_3DX,<name>_3DY,<name>_3DZ`); one row per frame; `0,0,0` when a joint is
absent. Values are `kps_3d` in metres.

### `.bvh` (`--bvh`)
Standard BVH motion-capture, one file per tracked person (`p_0.bvh`, …). Root
translation from `pred_cam_t` (×100 → cm), joint rotations decoded from
`global_rot` + `body_pose` + `hand_pose`, names mapped via `NAME_MAP`. See the
README **BVH export** section.

---

## 6. Coordinate spaces at a glance

| Output | Dim | Units | Frame |
|--------|-----|-------|-------|
| `yolo_kps` | 2D | pixels | original image |
| `kps_2d` | 2D | pixels | original image (projected) |
| `kps_3d` | 3D | metres | camera space (root-translated) |
| `pred_vertices` | 3D | metres | camera space |
| `.joints` / `.obj` | 3D | cm | BVH world (pelvis-relative, Y/Z flipped) |
| `.bvh` root | 3D | cm | BVH world |
