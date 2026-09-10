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

## Spec source

ARF is ISO/IEC 23090-39 (MPEG-I Part 39), at FDIS stage as of writing. This
implementation is based on the published system-level overview:

> J. Regateiro, A. Trioux, Q. Avril, "The MPEG Avatar Representation Format
> (ARF): An Interoperable Container and Animation Framework for Avatars,"
> *IEEE Computer Graphics and Applications*, 2026.
> https://ieeexplore.ieee.org/document/11667221

**This is not a certified-conformant ARF implementation.** The overview
article describes the JSON document's top-level structure, the ZIP/ISOBMFF
container options, and the shape of the Animation Stream Format (AAU header,
per-type sample fields, dense/sparse tensor formats) at a level of detail
sufficient to design against, but it is not the FDIS bitstream-syntax text
itself. Where the article doesn't pin down an exact bit layout (AAU type
numeric codes, the config-unit's exact field list, etc.), this writer makes
its own explicit choice and documents it below — treat those as **our
convention**, not verified spec values, until checked against the published
FDIS/IS text. `tools/validate_arf.py` is the authoritative description of
what this writer actually emits.

## Scope: what's implemented vs. skipped

Implemented:
- **Container**: ZIP-based `.arfz` only (the spec's simpler, more portable
  option — no ISOBMFF/MP4 muxing).
- **Base avatar model** (`arf.json`, hand-written JSON — see `arf_json.h`):
  `preamble`, `metadata`, `structure.animationStreams`, and `components` with
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
  (`MHR_LBS_Data::face_vectors`) as one packed dense tensor, animated by
  `MHRResult::face_params`.

Not implemented (out of scope for this writer; the spec defines all of
these):
- ISOBMFF container, RTP payload streaming, `MPEG_node_avatar` glTF scene
  integration, authentication/biometric features, protection/DRM
  configurations, landmark sets, texture sets/animation, LoDs, proprietary
  animation links, `AnimationLink`/`mapping` framework-conversion objects.
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
  when face export isn't enabled.
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
  ARF defines a sparse tensor MIME type for exactly this case.
- **Mesh and face-blendshape geometry are raw dense tensors**, not embedded
  glTF/GLB (the spec's "typically GLB" convention for `BlendshapeSets`
  targets) — this codebase has no glTF writer. Documented deviation; swap in
  real GLB targets if/when one exists.
- **Joint animation samples are 4×4 matrices**, per the spec's `AAU_JOINT`
  sample structure, composed each frame from the shared FK's local
  quaternion + translation + scale (`mhr_fk::State`) via `compose_trs_mat4()`
  (row-major `T · R · S`).
- **Timescale**: the config AAU sets `timescale = round(fps)` ticks/second;
  every AAU timestamp is the integer frame index (1 tick = 1 frame at the
  clip's fps) — avoids float drift.

## Container layout

```
<stem>_<id>.arfz                (ZIP, uncompressed-friendly — MZ_BEST_SPEED)
├── arf.json                    preamble, metadata, structure, components, data
├── data/
│   ├── mesh_positions.bin      dense tensor [n_verts, 3] float32
│   ├── mesh_indices.bin        dense tensor [n_tris, 3] uint32
│   ├── skin_weights.bin        sparse tensor, dims [n_verts, n_joints]
│   ├── inv_bind_pose.bin       dense tensor [n_joints, 16] float32 (row-major 4x4)
│   └── face_blendshapes.bin    dense tensor [72, n_verts, 3] float32 (only with --dev-face)
└── animations/
    ├── joints.bin               AAU_CONFIG + one AAU_JOINT per frame
    └── face.bin                 AAU_CONFIG + one AAU_BLENDSHAPE per frame (only with --dev-face)
```

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

### Avatar Animation Unit (AAU) framing (`append_aau`, `arf_writer.cpp`)

A simplified, byte-aligned version of the spec's header — **not** bit-exact
to the FDIS's 7-bit-packed `unit_type` field (see the honesty note above):

```
uint8   unit_type      0 = AAU_CONFIG, 1 = AAU_BLENDSHAPE, 2 = AAU_JOINT
uint32  unit_length     bytes of payload that follow
<payload>
```

`AAU_CONFIG` payload: `uint32 timestamp(0); uint32 profile_len; char[] profile; float32 timescale`.

`AAU_JOINT` payload: `uint32 timestamp_ticks; uint32 n_joints; { uint32 joint_index; float32[16] local_matrix } × n_joints`.

`AAU_BLENDSHAPE` payload: `uint32 timestamp_ticks; uint32 id_len; char[] target_blendshape_set_id; uint8 has_confidence(0); uint32 n_entries; { uint32 blendshape_index; float32 weight } × n_entries`.

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

python3 tools/validate_arf.py p_0.arfz
```

## Validating output

`tools/validate_arf.py` (stdlib `zipfile` + `json` only) is the reference
reader/smoke-test — unzips a `.arfz`, checks the mandatory `arf.json` keys,
cross-checks the skeleton's joint/parent structure, decodes the `AAU_JOINT`
stream and reports frame count / duration / the root joint's translation
range (useful for a quick sanity comparison against the corresponding
`--bvh` run's printed root-path range). There is no ARF viewer in this repo
or elsewhere available to this project, so this script — and manual
`unzip -l` / `python3 -m json.tool` inspection — is the practical way to
catch a malformed writer.

## Known limitations / future work

- `src/render/fast_sam_3dbody_render.cpp` (the live preview/debug renderer,
  which keeps its own local copy of the CLI flags rather than using
  `cli_common.h`) is not wired for `--arf`.
- Mesh/blendshape geometry is raw dense tensors rather than embedded glTF —
  see "Design decisions" above.
- No ISOBMFF container, RTP streaming, or scene-description integration.
- AAU binary framing is a documented simplification, not bit-exact to the
  FDIS text (which this project does not have access to).
