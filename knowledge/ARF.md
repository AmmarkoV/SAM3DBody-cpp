# ARF export (`--arf`)

`--arf PATH` writes one **MPEG Avatar Representation Format** container
(`.arfz`) per tracked person: `<stem>_<id>.arfz` (same per-person naming
convention as `--bvh`, including `--bvh-split-scenes`'s
`<stem>_scene<S>_person<P>` form — that flag applies to `--arf` output too).
Unlike `--bvh`, an ARF container is a **complete base avatar** — skeleton,
skin weights, a personalized rest mesh, and (optionally) facial blendshapes —
not just an animated skeleton, packaged with its own animation stream so the
whole thing is self-contained and loadable independently of this pipeline.

Implemented by `src/SAM3DBODY-cpp/arf_writer.{h,cpp}`
(+ `arf_json.{h,cpp}` for JSON serialization, `mhr_fk.{h,cpp}` for the shared
forward-kinematics core also used by `BVHWriter`). Wired into the offline
binary as Pass 7 (`offline::export_to_arf`, `offline_passes.cpp`) and into
the live binary (`fast_sam_3dbody_run`) alongside `BVHWriter`.

## Spec source and conformance status

ARF is ISO/IEC 23090-39 (MPEG-I Part 39). This writer was rewritten
2026-09-11 to match
[`AmmarkoV/ARFPlayer`](https://github.com/AmmarkoV/ARFPlayer)'s `libarf` — an
independent reader/writer whose component graph, `structure` Asset/LOD model,
`data[]` shape, and AAU animation-stream bitstream (field widths, big-endian
byte order) have been checked against the ISO/IEC 23090-39 **FDIS-stage
text** (see that project's `doc/CONFORMANCE_GAPS.md` for the full
accounting). Matching `libarf` is therefore one step closer to the actual
spec than this writer's original design, which was based only on the
published system-level overview:

> J. Regateiro, A. Trioux, Q. Avril, "The MPEG Avatar Representation Format
> (ARF): An Interoperable Container and Animation Framework for Avatars,"
> *IEEE Computer Graphics and Applications*, 2026.
> https://ieeexplore.ieee.org/document/11667221

**This is still not a certified-conformant ARF implementation** — neither
this writer's author nor `libarf`'s has the FDIS bitstream-syntax text open
side by side with every line below; `libarf`'s own conformance notes are
themselves the best available check. Verified end to end 2026-09-11 by
generating a real `.arfz` from this pipeline (`matrix.mp4`, 72 frames,
127 joints, 18439 vertices, 36874 triangles, 51337 skin nonzeros, 72 face
blendshapes) and loading it with `libarf`'s own reader (`arfinfo_cpp`,
`arfplay --info`) — skeleton, mesh, skin weights (weight-sum check passes),
body track and face track all decode correctly, and the container round-trips
through `libarf`'s own re-encoder unchanged. `tools/validate_arf.py` in this
repo was updated in lockstep with this rewrite and remains a quick,
dependency-free local smoke test; `libarf`'s own tools are the deeper,
independent check.

One deliberate, documented deviation remains from the spec, inherited
unchanged from before this rewrite: **skin weights use a sparse tensor
encoding**, not the dense form the spec defines (see "Design decisions"
below) — the same choice `libarf` itself makes and documents.

## Scope: what's implemented vs. skipped

Implemented:
- **Container**: ZIP-based `.arfz` only (the spec's simpler, more portable
  option — no ISOBMFF/MP4 muxing).
- **Base avatar model** (`arf.json`, hand-written JSON — see `arf_json.h`):
  `preamble`, `metadata`, `structure.assets[].lods[]`, and `components` with
  `nodes` (all 127 MHR joints), one `skeletons` entry, one `skins` entry, one
  `meshes` entry, and (when face export is on) one `blendshapeSets` entry.
- **Skeleton**: all 127 MHR joints (`src/SAM3DBODY-cpp/mhr_joint_table.h`),
  parent/child hierarchy, rest local translation + rotation per joint.
- **Skin**: sparse per-vertex joint weights (`MHR_LBS_Data::skin_*`) +
  inverse bind matrices.
- **Mesh**: a *personalized* rest mesh — `base_shape` plus this track's
  identity-shape coefficients (`MHRResult::shape`, averaged across the whole
  track and baked in once), not the generic template mesh — plus triangle
  topology from `body_mesh.tri`.
- **Animation stream**: per-frame `AAU_JOINT` samples (one 4×4 local
  transform per joint), and — only under `--dev-face` — per-frame
  `AAU_BLENDSHAPE` samples for facial expression.
- **Face blendshapes** (opt-in, `--dev-face`): the 72 facial PCA components
  (`MHR_LBS_Data::face_vectors`) as one minimal GLB file per component,
  animated by `MHRResult::face_params`.
- **`id_map.txt`**: a non-normative `<type>\t<id>\t<name>` debug sidecar,
  mirroring `libarf`'s own convention.

Not implemented (out of scope for this writer; the spec defines all of
these, and `libarf` already supports the writer side of the first three if a
real data source ever shows up in this pipeline):
- `LandmarkSet`/`AAU_LANDMARK` — no landmark tracking in this pipeline.
- `TextureSet`/`TextureTarget` — no textured-avatar export in this pipeline.
- LoDs beyond one (`structure.assets[].lods` always has exactly one entry).
- ISOBMFF container, RTP payload streaming, `MPEG_node_avatar` glTF scene
  integration, authentication/biometric features, protection/DRM
  configurations, proprietary animation links, `AnimationLink`/`mapping`
  framework-conversion objects.
- Identity-shape is **baked into the mesh**, not exposed as a live
  `BlendshapeSet` — see "Design decisions" below.

## Design decisions

- **Identity shape is baked, not streamed.** `MHRResult::shape` (45-dim) is
  roughly constant for one person across a track. Rather than exporting 45
  full-resolution shape-target meshes for coefficients that never change
  within the track, `dump_one_person()` averages `shape` across every frame
  of the track and applies it once to `base_shape` at close time, producing
  a personalized rest mesh. This matches ARF's "personalized 3D
  representation" framing of the base avatar model.
- **Facial expression is the one real animated `BlendshapeSet`.**
  `MHRResult::face_params` (72-dim) is zeroed by default
  (`PipelineConfig::zero_face_params`) unless `--dev-face` is passed. ARF
  export mirrors that gate exactly — the `face_expression` `BlendshapeSet`
  and its `AAU_BLENDSHAPE` track are omitted entirely (not just zero-filled)
  when face export isn't enabled. Each of the 72 face PCA basis vectors
  becomes one `shapes[i]` GLB, not 72 "named" facial expressions — this
  pipeline has no separate discrete-blendshape basis, only the PCA one.
- **Rest skeleton offsets are measured, not templated.** Each joint's rest
  `translation` in `arf.json` is the **median** rest-local bone vector
  observed across the whole track (`mhr_fk::State::rest_local_bone_vector`,
  sampled every frame in `ARFWriter::append_frame_for`), not the generic LBS
  rest offset — mirrors `BVHWriter::rewrite_offsets_for`'s
  measured-bone-length philosophy, so the exported avatar's proportions
  match the actual tracked person.
- **Root translation is the camera-translation head, not the FK.** The FK's
  own root joint (`body_world`, `joint_parents[0] == -1`) only ever carries a
  tiny internal wobble in its PT-decoded translation — actual world position
  comes from `MHRResult::pred_cam_t` (metres → the same ×100 cm convention
  `BVHWriter` uses). `ARFWriter::append_frame_for` special-cases the root's
  translation this way; every other joint's translation comes straight from
  the FK. (Verified: the exported root's per-frame translation range matches
  `BVHWriter`'s independently-computed root path to within float noise on a
  test clip.)
- **Skin weights use the sparse tensor encoding**, not the dense form the
  spec text illustrates first — `MHR_LBS_Data::skin_joint_idx/weights/vert_idx`
  are already a sparse COO-style list (51337 nonzeros over 18439×127), and
  ARF defines a sparse tensor MIME type for exactly this case. `libarf` makes
  the identical documented deviation for the same reason.
- **Mesh geometry is a raw dense tensor**; **blendshape-shape geometry is
  one minimal GLB per shape**, per spec — see "Container layout" below.
  `BlendshapeSet.shapes[i]`'s GLB stores that shape's **absolute** deformed
  vertex positions (base mesh + delta), not a delta: the spec's blend
  formula `v_out = v_0 + sum_i(w_i * (v_i - v_0))` computes the delta at
  blend time from the stored absolute position. `build_glb_mesh()` in
  `arf_writer.cpp` is a self-contained ~90-line encoder (one mesh, one
  primitive, a `POSITION` accessor and an indices accessor, no materials or
  textures) — ported from `libarf`'s own `arf_glb.c` reference encoder.
- **Component ids are numeric**, resolved by matching value against the
  target array's own declared `id` field, per the spec's General
  Conventions clause ("index-based referencing is not used in this
  specification"). This writer always assigns sequential `id == index` for
  everything it writes, since it only ever holds one mesh/skin/skeleton/
  blendshape set — a reader must still resolve by id, not assume it, but
  every reference below can be read as "the only one" without ambiguity.
  `data[]` ids are a running count: `mesh_positions`=0, `mesh_indices`=1,
  `skin_weights`=2, `inverse_bind`=3, then one id per blendshape shape (only
  present with `--dev-face`) — see `ARFWriter::dump_one_person`.
- **Joint animation samples are 4×4 matrices**, per the spec's `AAU_JOINT`
  sample structure, composed each frame from the shared FK's local
  quaternion + translation + scale (`mhr_fk::State`) via `compose_trs_mat4()`
  (row-major `T · R · S`). **An AAU local matrix is the complete local
  transform** — do not additionally apply the node's rest translation/
  rotation when animating (those exist for drawing the rest pose only).
- **Timescale**: the config AAU sets `timescale = round(fps)` ticks/second;
  every AAU timestamp is the integer frame index (1 tick = 1 frame at the
  clip's fps) — avoids float drift.
- **Node/skin/mesh `mapping`/`path`** are honest placeholders (the node's
  own name, or a fixed string like `"mesh0"`/`"skin0"`) — this project has
  no verified taxonomy for the semantic scene-graph paths the spec's
  companion scene-description part (23090-14) defines, matching `libarf`'s
  own convention.

## Coordinate convention

**Y-up, −Z forward, centimetres, origin at the camera.** A reader can display a
container directly: no basis change, no axis flip, no unit scale.

- The rest mesh and the inverse bind matrices are Y-up (feet near `Y=0`, head
  near `Y=+173`), and the animated pose keeps that — head above feet in `+Y`.
- The root node's per-frame translation is `(+tx, −ty, −tz) · 100` from
  `pred_cam_t`. The **negation is deliberate**: `pred_cam_t` is camera-space
  (+X right / +Y *down* / +Z forward), and the MHR renderer reconciles it with
  the Y-up pose via `display = v_model + F·t`, `F = diag(1,−1,−1)`
  (`mhr_pose_driver.h:252-271`; the LBS buffer's `verts[Y,Z] *= -1` and the view
  matrix's flip cancel on the vertices, so only the translation carries `F`).
- Consequently a standing subject straddles the origin: feet around `Y ≈ −100`
  cm, head around `Y ≈ +60` cm for a camera held at ~1.2 m, at `Z ≈ −200…−350`
  cm — in front of a camera that looks down `−Z`.

Exports before 2026-09-11 wrote `+t` on all three axes, which put the body
*above* the camera and *behind* it in Z, and mirrored its vertical motion.
`tools/validate_arf.py` now asserts the placement, so a regression is caught at
validation time. `bvh_writer.cpp` carries the same convention (its
`gmr_retarget.py --flip-depth` compensation is retired — see `GMR.md` §6).

## Container layout

```
<stem>_<id>.arfz                (ZIP, uncompressed-friendly — MZ_BEST_SPEED)
├── arf.json                    preamble, metadata, structure, components, data
├── id_map.txt                  non-normative id -> name debug index
├── data/
│   ├── mesh_positions.bin      dense tensor [n_verts, 3] float32
│   ├── mesh_indices.bin        dense tensor [n_tris, 3] uint32
│   ├── skin_weights.bin        sparse tensor, dims [n_verts, n_joints]
│   ├── inv_bind_pose.bin       dense tensor [n_joints, 16] float32 (row-major 4x4)
│   └── face_blendshape_<i>.glb GLB, one per face PCA component (only with --dev-face)
└── animations/
    ├── joints.bin               AAU_CONFIG + one AAU_JOINT per frame
    └── face.bin                 AAU_CONFIG + one AAU_BLENDSHAPE per frame (only with --dev-face)
```

Every integer/float in the dense/sparse tensor payloads and inside each GLB
is **little-endian** (this project's own convention, matching `libarf`'s).
The AAU animation stream is the one exception: it is **big-endian** — see
below.

### Dense tensor binary layout (`build_dense_tensor`, `arf_writer.cpp`)

```
int32   num_of_dims
int32[] dims               (num_of_dims entries)
int32   dtype               glTF component-type code (5126 = FLOAT, 5125 = UNSIGNED_INT)
<raw row-major data, dims[0]*dims[1]*...*component_size bytes>
```

### Sparse tensor binary layout (`build_sparse_tensor`)

```
int32   num_of_dims
int32[] dims
int32   valueCount
int32   itype                (5125 = UNSIGNED_INT — index component type)
int32   dtype                (5126 = FLOAT — value component type)
uint32[valueCount]  flat row-major indices  (index = v * dims[1] + j, i.e. vertex*n_joints + joint)
float32[valueCount] values
```

### `BlendshapeSet.shapes[i]`: one minimal GLB per target (`build_glb_mesh`)

A self-contained binary glTF with only geometry: one buffer, two
bufferViews (positions, indices), two accessors (`POSITION` FLOAT VEC3,
indices UNSIGNED_INT SCALAR), one mesh/primitive, no materials or textures.
Positions are that shape's **absolute** deformed vertices (base mesh +
delta); indices are byte-identical to `mesh_indices.bin`'s topology, per the
spec's "topology of the baseMesh and the associated shapes shall be
identical" requirement.

### Avatar Animation Unit (AAU) framing (`append_aau`, `arf_writer.cpp`)

**Big-endian**, matching the spec's `uimsbf` bitstream-table convention (the
same one ISOBMFF/MPEG-2 Systems use for that notation):

```
uint8   (unit_type << 1) | reserved     0=AAU_CONFIG, 1=AAU_BLENDSHAPE, 2=AAU_JOINT
uint32BE unit_length                    payload bytes that follow
<payload, all fields big-endian>
```

`AAU_CONFIG` payload: `uint32BE timestamp(0); uint8 profile_length;
byte[profile_length] profile` (no NUL); `float32BE timescale`.

`AAU_JOINT` payload: `uint32BE timestamp_ticks; uint16BE joint_set_id;
uint8 (velocity_present<<7)|reserved; uint16BE joint_count_minus1;
{ uint16BE joint_index; float32BE local_matrix[16] } × (joint_count_minus1+1)`.
`joint_set_id` is `components.skeletons[0].id` (always 0). This writer never
has velocity data, so `velocity_present` is always 0.

`AAU_BLENDSHAPE` payload: `uint32BE timestamp_ticks; uint16BE
blendshape_set_id; uint8 (confidence_present<<7)|reserved; uint16BE
blendshape_count_minus1; { uint16BE blendshape_index; float32BE weight } ×
(blendshape_count_minus1+1)`. `blendshape_set_id` is
`components.blendshapeSets[0].id` (always 0); `confidence_present` is
always 0.

## CLI

```
--arf PATH        Write MPEG ARF avatar container(s) (.arfz); per-person filenames
                   appended, same convention as --bvh.
--dev-face         Enable face expression params — also gates the ARF face
                   BlendshapeSet + AAU_BLENDSHAPE stream (off by default in both
                   the live and offline binaries).
```

`--arf` is parsed in `src/SAM3DBODY-cpp/cli_common.h` (shared by every
binary that includes it) next to `--bvh`. It's independent of `--bvh` — pass
either, neither, or both; `--from` plus at least one of `--bvh`/`--arf` is
required by the offline binary.

## Example

```bash
./build/offline_sam_3dbody_render --from clip.mp4 --arf ./p.arfz --dev-face
# → p_0.arfz  (one per tracked person)
```

## Validating output

```bash
python3 tools/validate_arf.py p_0.arfz --verbose   # dependency-free local smoke test
```

For a deeper, independent check, `AmmarkoV/ARFPlayer`'s own tools read the
exact shape this writer produces and have been checked against the
ISO/IEC 23090-39 FDIS-stage text:

```bash
/path/to/ARFPlayer/build/arfinfo_cpp p_0.arfz            # skeleton/mesh/skin/frame summary
/path/to/ARFPlayer/build/arfplay --info p_0.arfz          # + face/landmark/texture track detail
```

## Known limitations / future work

- `src/render/fast_sam_3dbody_render.cpp` (the live preview/debug renderer,
  which keeps its own local copy of the CLI flags rather than using
  `cli_common.h`) is not wired for `--arf`.
- No ISOBMFF container, RTP streaming, or scene-description integration.
- Skin weights stay on the sparse encoding rather than the spec's dense
  form — see "Design decisions" above.
- No `LandmarkSet`/`TextureSet` export — nothing in this pipeline produces
  landmark or texture data yet; `libarf` already supports reading/writing
  both if that changes.
