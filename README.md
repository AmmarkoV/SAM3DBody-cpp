# SAM3DBody-cpp

Standalone C++ inference engine for **SAM-3D-Body** — zero Python dependency at runtime.

Takes a BGR image and produces per-person MHR body pose parameters, camera translation, and optionally full 3D mesh vertices + 70 body/hand keypoints, all via ONNX Runtime + ggml.

Also includes Python frontends that call the compiled shared library via ctypes, and a CSV exporter for the 70 MHR keypoints.

### 🎬 Multi-person BVH motion-capture export

`--bvh PATH` writes a **standard BVH motion-capture file per detected person** (`p_0.bvh`, `p_1.bvh`, …).
Identities are kept stable across frames by a built-in 2D-bbox IoU tracker, each
file's joint OFFSETs are auto-resized to the actor's measured bone lengths, and the
output drops straight into Blender / BVHTester / any DCC. A bundled
[`blender/blender_bvh_plugin.py`](blender/blender_bvh_plugin.py) drives a
MakeHuman-rigged character from the result. See **[BVH export](#bvh-export---bvh)** for details.

```bash
./fast_sam_3dbody_run --from clip.mp4 --bvh ./p.bvh --headless
# → p_0.bvh, p_1.bvh, …
```

![SAM3DBody-cpp](doc/screen.jpg)

---

## Models

Pre-built ONNX / GGUF / LBS model files are hosted on HuggingFace:

**[https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models)**

### With a CUDA GPU (recommended)

Download the all-in-one zip, extract it, and place the resulting `onnx/` directory at the repo root:

```bash
wget https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-onnx-models.zip
unzip SAM3DBody-cpp-onnx-models.zip
# onnx/ is now at the repo root — ready to build
```

| File | Size | Description |
|------|------|-------------|
| `onnx/backbone.onnx` + `.data` | ~4.8 GB | DINOv3-ViT-H/14+ encoder (BF16, CUDA EP only) |
| `onnx/decoder.onnx` | ~93 MB | 6-layer PromptableDecoder |
| `onnx/yolo.onnx` | ~81 MB | YOLO11m-pose person detector |
| `onnx/pipeline.gguf` | ~5 MB | MHR + camera projection heads |
| `onnx/body_model.lbs` | ~27 MB | Native C LBS data (joints, weights, shape) |
| `onnx/correctives.bin` | ~33 MB | Pose corrective blend shapes |
| `onnx/keypoint_mapping.bin` | ~8 KB | MHR-70 keypoint index map |

### Without a CUDA GPU (CPU-only)

The standard `backbone.onnx` is exported in BFloat16 and **requires a CUDA GPU** — it will not load on the ORT CPU execution provider.
A float32 CPU-compatible backbone is available as a separate download from the same HuggingFace repo.
 

Download these two files and place them alongside the rest of the models in `onnx/`:

| File | Size | Description | Links |
|------|------|-------------| ---- |
| `backbone_fp32.onnx` | ~1 MB | Graph (references external data) | [Link](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/blob/main/backbone_fp32.onnx) |
| `backbone_fp32.onnx.data` | ~3.2 GB | Float32 weights — no BF16, CPU EP compatible | [Link](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/blob/main/backbone_fp32.onnx.data) |

Then run with `--backbone backbone_fp32.onnx --cuda -1`:

```bash
./build/fast_sam_3dbody_run \
    --onnx-dir ./onnx \
    --backbone backbone_fp32.onnx \
    --cuda     -1 \
    --from     your_video.mp4
```

> **Performance expectations for CPU inference:**
> DINOv3-ViT-H has ~630 M parameters. On a modern laptop CPU, one backbone
> forward pass takes **5–15 seconds**, making video processing impractical.
> Single-image or low-frequency use cases are feasible. For anything
> approaching real-time, a CUDA-capable GPU is required.
>
> WSL2 users: if `nvidia-smi` works inside WSL, install the
> [CUDA toolkit for WSL2](https://developer.nvidia.com/cuda-downloads)
> (do **not** install the full Linux driver — only the WSL2 toolkit) and use
> the standard `backbone.onnx` instead.

> **CMake will warn** at configure time if neither `onnx/` nor the zip is found.

---

## Pipeline

This is **not** 2D-to-3D lifting. The network directly regresses 3D body model
parameters from image features — no depth sensor, no floor plane, no stereo.

Starting from a raw image the pipeline does:

1. **YOLO** detects person bounding boxes.
2. Each crop is fed to a **DINOv2-ViT-H backbone** producing a `[1280, 32, 32]`
   spatial feature map.
3. A **transformer decoder** (conditioned on the crop's ray directions and
   focal length) compresses that feature map into a 1024-dim pose token.
4. Two small **FFN heads** (run on CPU via ggml) decode the token into:
   - **519 pose parameters** — global orientation (6D continuous rotation),
     per-joint Euler angles for 127 joints, SMPL-like shape betas (45),
     hand pose (108), and face expression (72).
   - **3 camera parameters** `[scale, tx, ty]` → world-space translation
     `[tx, ty, tz]`.
5. Those parameters drive **linear blend skinning (LBS)** over 18 439 vertices
   to produce the full body mesh and 70 keypoints.

The focal length is estimated from the image diagonal and baked into the
decoder's conditioning input, so the network learns to associate apparent body
size in the crop with metric depth — the same monocular depth-from-body-proportions
approach used by SMPL/HMR-family models. The body is placed in **camera space**
via the predicted root translation; floor-plane recovery is a downstream step if
needed.

```
BGR image
  │
  ▼  yolo.onnx            ONNX Runtime (CUDA EP)   person bboxes + 17 COCO keypoints
  │
  ▼  backbone.onnx        ONNX Runtime (CUDA EP)   feature map  [B, 1280, 32, 32]
  │   DINOv3-ViT-H/14+
  │
  ▼  decoder.onnx         ONNX Runtime (CUDA EP)   pose token   [B, 1024]
  │   6-layer PromptableDecoder
  │
  ▼  pipeline.gguf        CPU matmul (ggml)         MHR params [B, 519] + camera [B, 3]
  │   MHR head + camera head weights
  │
  ▼  body_model.lbs       native C LBS (optional)   vertices [18439, 3] in metres
      extracted once by tools/extract_lbs_data.py
```

> **Note:** `body_model.onnx` export is blocked on PyTorch ≥ 2.x (torch.export rejects
> TorchScript modules). The native C LBS path reads `body_model.lbs` directly and
> produces identical output to Python `mhr_forward` (body model stores data in cm;
> `mhr_lbs_compute` applies ×0.01 to match Python's `/100` conversion).

**Per-person output** (`MHRResult` / `FsbResult`):

| Field | Shape | Description |
|-------|-------|-------------|
| `bbox` | [4] | x1 y1 x2 y2 in original image pixels |
| `focal_length` | scalar | Estimated focal length (pixels) |
| `pred_cam_t` | [3] | Raw camera head output: [s, tx, ty] |
| `global_rot` | [3] | Global orientation – Euler ZYX (radians) |
| `body_pose` | [133] | Body joint angles – Euler |
| `shape` | [45] | SMPL-like identity blend shape betas |
| `scale` | [28] | Scale PCA components |
| `hand_pose` | [108] | Hand joints: left [54] + right [54] |
| `face_params` | [72] | Facial expression parameters |
| `mhr_model_params` | [204] | Assembled LBS parameter vector (passed to `mhr_lbs_compute`) |
| `yolo_kps` | [51] | COCO 17 keypoints × [x, y, confidence] |
| `pred_vertices` | [55317] | 18439 verts × 3, metres (when native C LBS runs) |
| `kps_3d` | [210] | 70 joints × 3, metres (when native C LBS runs) |
| `kps_2d` | [140] | 70 joints × 2 projected (when native C LBS runs) |

---

## Directory layout

```
SAM3DBody-cpp/
├── CMakeLists.txt
├── body_mesh.tri                     SMPL-like body mesh for the GL renderer
├── fast_sam_3dbody_frontend.py       Python lightweight frontend (ctypes, no extra deps)
├── fast_sam_3dbody_frontend-3D.py    Python 3D frontend (ctypes + Python body model)
├── fast_sam_3dbody_dump_csv.py       Python CSV exporter – 70 MHR keypoints per frame
├── two_pass.py                       Second-pass temporal smoother
├── ros_demo_webcam.py                ROS demo
├── onnx/                             Runtime model files – download from HuggingFace (see above)
│   ├── backbone.onnx + .data         ~4.8 GB  DINOv3-ViT-H/14+ encoder
│   ├── decoder.onnx                  ~93 MB   6-layer PromptableDecoder
│   ├── pipeline.gguf                 ~5 MB    MHR + camera heads
│   ├── yolo.onnx                     ~81 MB   YOLO11m-pose
│   ├── body_model.lbs                ~27 MB   native C LBS data
│   ├── correctives.bin               ~33 MB   pose corrective blend shapes
│   └── keypoint_mapping.bin          ~8 KB    MHR-70 keypoint index map
├── GraphicsEngine/
│   ├── System/glx3.{h,c}            GLX window management
│   └── ModelLoader/                  .tri mesh loader + LBS joint transform
├── AmMatrix/                         Lightweight C matrix / quaternion library
├── render/
│   ├── fast_sam_3dbody_render.cpp    OpenGL mesh overlay renderer
│   └── mhr_pose_driver.h             LBS driver (camera matrices, vertex update)
├── scripts/
│   └── build.sh / setup.sh / webcam.sh / video.sh / offline_video.sh
└── src/
    ├── fast_sam_3dbody.h             C++ public API
    ├── fast_sam_3dbody.cpp           Pipeline implementation
    ├── fast_sam_3dbody_capi.h        Plain C API (for ctypes)
    ├── fast_sam_3dbody_capi.cpp
    ├── preprocess.hpp                Crop, normalise, ray_cond, NMS, pose conversion
    ├── bvh_writer.h / bvh_writer.cpp BVH motion-capture exporter (multi-person, name-mapped to MHR joints)
    ├── mhr_joint_table.h             Generated by scripts/build_joint_table.py — MHR joint names + parents
    └── main.cpp                      CLI executable (--bvh, --out CSV, live overlay window)
```

> **Developer tools** (ONNX/GGUF export, LBS extraction, debug scripts, Python training env) live in the parent project:
> **[https://github.com/AmmarkoV/Fast-SAM-3D-Body](https://github.com/AmmarkoV/Fast-SAM-3D-Body)**

---

## Setup

### 1. Download models

See the **[Models](#models)** section above. After extracting the zip, `onnx/` should be at the repo root.

> To build models from source (requires the original Python training environment), see
> [Fast-SAM-3D-Body](https://github.com/AmmarkoV/Fast-SAM-3D-Body).

### 2. Build

Requirements: CMake ≥ 3.18, C++17 compiler, OpenCV (core/imgproc/videoio/highgui/dnn), optional CUDA Toolkit.

```bash
cd fast_sam_3dbody_cpp
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

CMake handles dependencies automatically:
- **ONNX Runtime 1.20.1** – downloaded from GitHub releases if not found; point to an existing install with `-DONNX_RUNTIME_DIR=/path/to/onnxruntime`
- **ggml** – fetched via `FetchContent` from GitHub
- **CUDA** – auto-detected; set `-DCMAKE_CUDA_ARCHITECTURES=86` (or `75`, `89`, etc.) for your GPU; falls back to CPU-only if not found

Outputs in `build/`:

| File | Description |
|------|-------------|
| `fast_sam_3dbody_run` | Standalone CLI executable |
| `libfast_sam_3dbody.so` | Shared library for C++ linking or ctypes |

---

## Running

### CLI executable

```bash
cd fast_sam_3dbody_cpp/build

# Single image – prints pose params to stdout
./fast_sam_3dbody_run \
    --onnx-dir ../onnx \
    --gguf     ../onnx/pipeline.gguf \
    --yolo     ../onnx/yolo.onnx \
    --from     ../../assets/teaser.png

# Webcam (device 0)
./fast_sam_3dbody_run \
    --onnx-dir ../onnx --gguf ../onnx/pipeline.gguf --yolo ../onnx/yolo.onnx \
    --from 0

# Video file
./fast_sam_3dbody_run \
    --onnx-dir ../onnx --gguf ../onnx/pipeline.gguf --yolo ../onnx/yolo.onnx \
    --from /path/to/video.mp4

# Fastest mode – skip LBS body model (no vertices, just pose params)
./fast_sam_3dbody_run ... --skip-body

# CPU-only
./fast_sam_3dbody_run ... --cuda -1
```

Full option list:

```
--onnx-dir PATH    Directory with backbone/decoder/body_model ONNX files
--gguf     PATH    pipeline.gguf (MHR + camera heads)
--yolo     PATH    YOLO pose model (.onnx)
--from     SRC     Webcam index (0,1,..) or path to image/video
-o / --out PATH    Write 70-joint 3D keypoints to CSV per frame
--bvh      PATH    Write BVH motion capture file(s) to PATH (see "BVH export" below)
--bvh-template P   BVH skeleton template (default: ./body.bvh)
--no-bvh-body-shape-change   Keep template body bone lengths (skip per-person rewrite)
--no-bvh-hand-shape-change   Keep template hand/finger bone lengths (skip per-person rewrite)
--bvh-raw-fingers            Do NOT rescale finger End-Site OFFSETs (keeps body.bvh's authored fingertips)
--cuda     DEVICE  CUDA device index (default 0; -1 = CPU)
--skip-body        Skip body model (no vertices / keypoints)
--headless         No display window
--thresh   T       YOLO person confidence threshold (default 0.50)
--nms      T       YOLO NMS IoU threshold (default 0.45)
--fx / --fy F      Camera focal length x/y in pixels (0 = image width)
--cx / --cy F      Principal point (0 = image centre)
--render-size W H  Override display window size
--size W H         Webcam capture resolution
--fps Z            Webcam capture framerate
--butterworth      Apply Butterworth low-pass filter to MHR output vectors
--bw-cutoff HZ     Butterworth cutoff frequency in Hz (default 6.0)
--butterworth-root-rotation  Filter global_rot in quaternion space (1st-order SLERP-EMA, see "Output filtering")
--rot-clamp DEG    Geodesic SLERP-step clamp on global_rot in degrees/frame (default 1; 0 = no clamp)
--info             Print pipeline info and exit
--help             Show this message
```

### Python lightweight frontend

Draws COCO 2D skeletons and a pose-bar panel. Requires only `opencv-python` and `numpy` — no PyTorch.

```bash
# From the repo root:
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from assets/teaser.png

# Webcam, cap at 3 persons
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py --from 0 --max-skeletons 3

# Save output image / video
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py \
    --from assets/teaser.png --out out.jpg

python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend.py \
    --from video.mp4 --headless --out out.mp4
```

Key options (same as CLI, plus):

```
--max-skeletons N  Cap persons drawn per frame (0 = unlimited)
--headless         No display window
--out PATH         Write result to image or video file
```

### Python 3D frontend

Full 3D mesh rendering identical to `demo_webcam.py`: four-panel output
`[original | 2D skeleton | front mesh | side mesh]`.

Uses the C engine for the fast path (YOLO → backbone → decoder → MHR FFN heads),
then calls the Python MHR body model (`mhr_model.pt`) for LBS skinning to produce
mesh vertices. Requires the full Python environment (PyTorch, sam_3d_body package, pyrender).

```bash
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py --from assets/teaser.png

# Webcam
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py --from 0 --max-skeletons 3

# Custom checkpoint paths
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py \
    --from assets/teaser.png \
    --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr-model  ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# Save result
python fast_sam_3dbody_cpp/fast_sam_3dbody_frontend-3D.py \
    --from assets/teaser.png --out result_3d.jpg
```

Key options (same as lightweight frontend, plus):

```
--checkpoint PATH  Path to model.ckpt (default: checkpoints/sam-3d-body-dinov3/model.ckpt)
--mhr-model  PATH  Path to mhr_model.pt
--device     STR   PyTorch device for body model: cuda or cpu (default: auto)
```
 

---

## Output filtering

The CLI can apply a second-order [Butterworth low-pass filter](https://en.wikipedia.org/wiki/Butterworth_filter)
to the MHR output vectors on every frame, reducing per-frame jitter without introducing ripple in the
passband.

```bash
# Enable with the default 6 Hz cutoff
./fast_sam_3dbody_run --from video.mp4 --butterworth

# Lower cutoff for smoother (more lag) output
./fast_sam_3dbody_run --from video.mp4 --butterworth --bw-cutoff 3.0

# Higher cutoff to preserve faster motion
./fast_sam_3dbody_run --from video.mp4 --butterworth --bw-cutoff 10.0
```

The filter is applied in-place to each detected person's result immediately after inference,
so all downstream consumers (CSV writer, BVH writer, display overlay) receive filtered data.

### Filtered vectors

| Vector | Channels | Method | Description |
|--------|----------|--------|-------------|
| `keypoints_3d` | 210 (70 joints × 3) | Butterworth | 3-D joint positions in metres |
| `body_pose` | 133 | Butterworth | Body joint Euler angles |
| `hand_pose` | 108 | Butterworth | Hand joint angles (left 54 + right 54) |
| `global_rot` | 3 | **Quaternion SLERP-EMA** | Global orientation – filtered on SO(3), see below |
| `pred_cam_t` | 3 | Butterworth | Camera / root translation |

`global_rot` is **not** filtered with the scalar Butterworth path. Doing that
on either the Euler triple or the four quaternion components fails for two
reasons rooted in the geometry of rotations:

1. **Euler wrap.** A rotation passing 180° jumps from +π to −π on the
   axis storage, even though the actual orientation moved 0°. A per-axis
   low-pass interpolates *linearly through that discontinuity*, briefly
   flipping the whole body for ~τ seconds (the filter time-constant) every
   time the model crosses a wrap or a gimbal-lock pole.
2. **SO(3) is not a vector space.** Butterworth's "maximally-flat magnitude"
   guarantee is a theorem about scalar LTI systems; it does not transfer to
   rotations no matter which parametrisation you filter. Per-channel
   smoothing of an Euler triple produces a composed rotation that is neither
   the input nor a geometrically meaningful interpolant.

So instead, when `--butterworth-root-rotation` is on, `global_rot` is filtered
in **quaternion space** by a 1st-order SLERP-EMA (`QuatLPF` in
`src/outputFiltering.h`):

  1. Convert the input Euler triple to a unit quaternion.
  2. **Hemisphere-correct** — if `dot(q_filt_prev, q_input) < 0`, negate
     `q_input`. `q` and `−q` represent the same orientation, so this removes
     the spurious 180° jump.
  3. Compute the geodesic angle `θ = 2·acos(|dot|)` and pick a SLERP
     fraction `t = min(α, --rot-clamp / θ)`. The `α` comes from the same
     `--bw-cutoff` time-constant used elsewhere.
  4. `q_filt ← SLERP(q_filt_prev, q_input_hemi, t)`, renormalise.
  5. Convert back to Euler-ZYX for storage.

This trades Butterworth's maximally-flat magnitude (a guarantee that does
not apply to SO(3) anyway) for **geodesic monotonicity**, **no-flip
continuity** across the wrap, and a single rotation-angle distance metric.
It's 1st-order (−6 dB/octave) where the linear channels are 2nd-order, but
that's the right trade-off for rotation — see the comment block in
`outputFiltering.h` for the full rationale.

`--rot-clamp` is now a geodesic outlier *clamp on the SLERP step*, not a
frame-rejection threshold: rapid input motion gets attenuated to at most
`rot-clamp` deg/frame in the output, never frozen. A single bad FFN frame
with a 180° flip is therefore absorbed smoothly across a few frames rather
than overwriting the orientation.

### Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--butterworth` | off | Enable the filter |
| `--bw-cutoff HZ` | `6.0` | Cutoff frequency in Hz. Lower = smoother but more temporal lag. Human motion typically stays below 6 Hz; use 3–4 Hz for very smooth output, 8–10 Hz to preserve fast gestures. |
| `--butterworth-root-rotation` | off | Run the quaternion SLERP-EMA on `global_rot` (see above). Off by default because the very first input frame seeds the filter, so a bad initial prediction would otherwise lock the sequence to it. |
| `--rot-clamp DEG` | `1.0` | Geodesic clamp on the SLERP step in **degrees per frame**. The filter's output never moves more than this many degrees of rotation per frame. With the new quaternion filter this *attenuates* fast input rather than rejecting it. Set to `0` to disable the clamp (pure EMA). |

The sampling rate is taken from `--fps` when specified, otherwise 30 Hz is assumed.
Each person slot maintains its own independent filter bank; new person slots are
initialised with a warm-up pass on their first frame so the filter starts from the
measured value rather than zero.

### Implementation

Implemented in `src/outputFiltering.h` — a header-only, dependency-free,
C-compatible file containing two primitives:

- `ButterWorth` — scalar 2nd-order IIR low-pass used for `keypoints_3d`,
  `body_pose`, `hand_pose`, `pred_cam_t`. Initialised with
  `initButterWorth(sensor, fs, fc)` and stepped with `filter(sensor, value)`.
- `QuatLPF` — 1st-order SLERP-EMA on unit quaternions, used only for
  `global_rot`. Initialised with `init_quat_lpf(&qf, fs, fc)` and stepped
  with `filter_quat(&qf, in_quat, max_step_rad, out_quat)`. The header also
  provides `euler_zyx_to_quat(rz, ry, rx, out)` and `quat_to_euler_zyx(in,
  &rz, &ry, &rx)` since `global_rot` is stored in MHR's ZYX-intrinsic Euler
  order.

---

## BVH export (`--bvh`)

Exports the per-frame MHR pose as one or more standard BVH motion-capture files.
The hierarchy is taken from a BVH template (default `./body.bvh`, a [MocapNET](https://github.com/FORTH-ModelBasedTracker/MocapNET)/[MakeHuman](https://static.makehumancommunity.org/)
T-pose skeleton); the motion comes from the MHR pipeline.

```bash
# Single image / video / webcam → one or more <name>_<id>.bvh files
./fast_sam_3dbody_run --from boom.mp4 --bvh ./p.bvh --headless
# produces ./p_0.bvh, ./p_1.bvh, … (one per tracked person)
```

### What gets written

For each detected person, every frame:

* **Root joint** — translation from `pred_cam_t` (×100 to convert metres → cm)
  and orientation from MHR's global rotation, in the BVH root's
  `Zrotation Yrotation Xrotation` channel order.
* **Body joints** matched by name to MHR (~50 joints incl. spine, arms, legs,
  fingers — full table in `src/bvh_writer.cpp` `NAME_MAP`). The local rotation
  is computed in MHR's frame as `inv(delta[parent]) · delta[self]`, where
  `delta[j] = R_global_mhr[j] · R_global_mhr_rest[j]⁻¹`, then decomposed to the
  joint's BVH channel order (`Zrotation Xrotation Yrotation`).
* **Unmapped BVH joints** (toes, face details, BVH metacarpals, etc.) stay at
  zero rotation — MHR doesn't predict angles for them.
* **OFFSETs** are rewritten at close-time to each person's median observed bone
  length so the template T-pose proportions match the actual subject. Bone
  *direction* is preserved (changing it disrupts the T-pose look the BVH file
  was authored with).
* **Per-category opt-out** with `--no-bvh-body-shape-change` and
  `--no-bvh-hand-shape-change`. body.bvh's authored hands are shorter than the
  MHR skeleton's (proximal phalanx 2.3 cm vs 3.8 cm), so rewriting visibly
  *enlarges* fingers — pass `--no-bvh-hand-shape-change` to keep body.bvh's
  authored hand proportions while still resizing the rest of the body to match
  the subject. The body flag is the symmetric escape hatch for the trunk/limbs.
* **Finger-tip End-Site compensation** (default on). The End-Site OFFSETs at
  the end of each finger have no channels so the bone-length rewrite never
  touches them, and body.bvh's are ~0.3 cm longer than the MHR `*_null` tip
  extensions. By default we rescale all 10 finger End-Site OFFSETs to the
  MHR length (direction preserved). Pass `--bvh-raw-fingers` to leave them
  exactly as authored.

### Multi-person export

Multiple skeletons in the same scene are exported as **separate BVH files**, one
per tracked identity. Identity persistence is handled by a built-in
bbox-IoU greedy tracker:

* IoU threshold `0.10`; tracks are retired after `90` frames missing (≈ 3 s at
  30 fps).
* Detections with degenerate bboxes (anchored at `(0, 0)` or near-zero area)
  are dropped before they hit the tracker — these are common YOLO failure
  modes that otherwise would spawn spurious tracks.
* While a track is alive but missing this frame, its previous pose is
  duplicated so the BVH timeline stays continuous through brief occlusions.

Filenames are derived from `--bvh PATH`:

| `--bvh` value | Outputs                              |
|---------------|--------------------------------------|
| `p.bvh`       | `p_0.bvh`, `p_1.bvh`, …              |
| `out/run`     | `out/run_0.bvh`, `out/run_1.bvh`, …  |
| `cap.mocap`   | `cap_0.mocap`, `cap_1.mocap`, …      |

Each file is fully independent — drop into Blender / BVHTester / any DCC.

### Validating the output

```bash
# Render a 3D-keypoints CSV at the same time, then compare hip-relative
# joint positions between the BVH and MHR's own 3D keypoints.
./fast_sam_3dbody_run --from boom.mp4 \
    --bvh ./p.bvh --out /tmp/boom_mhr.csv --headless

source venv/bin/activate
python3 scripts/verify_bvh_motion.py ./p_0.bvh /tmp/boom_mhr.csv
```

A clean run prints per-joint median / p90 / max error in cm; numbers around
2–5 cm on trunk joints and 5–15 cm on extremities are the expected range
(the larger residual on hands/feet partly reflects MHR's 70-keypoint surface
landmarks not coinciding with the LBS rotation-centre joints).

### How the mapping is built

The MHR body model uses 127 named joints (`body_world`, `root`, `l_uparm`,
`r_thumb1`, …) — names that don't appear in `body_model.lbs` but are exported
from the JIT model:

```bash
# (Re)generate src/mhr_joint_table.h from a checkpoint
source venv/bin/activate
python3 scripts/build_joint_table.py
# pass MHR_MODEL_PT=/path/to/mhr_model.pt to override the default location
```

`bvh_writer.cpp` then matches each BVH joint name to an MHR name via the
hand-authored `NAME_MAP` table. To support a new BVH template just add entries
to that table (and rebuild).

### Implementation notes

* The BVH I/O comes from a vendored, trimmed-down copy of
  [`MotionCaptureLoader`](https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/MotionCaptureLoader)
  in `GraphicsEngine/MotionCaptureLoader/` (plus `TrajectoryParser/InputParser_C`).
  We use `bvh_loadBVH` for parsing the template hierarchy and `dumpBVHToBVH` for
  serialising the result.
* The per-person motion buffer lives in `std::vector<float>` and is transplanted
  into `mc->motionValues` at close-time, then detached before `bvh_free()` so
  the library doesn't try to free our std::vector storage.
* If you only have one person in the scene you'll get exactly one file
  (`<name>_0.bvh`) — there is no single-file mode for backwards compatibility.

### Driving a MakeHuman model in Blender

`blender/blender_bvh_plugin.py` is a Blender add-on (by [AmmarkoV](https://github.com/AmmarkoV)) that
plays one of these BVHs onto a MakeHuman-rigged character via the
[**mpfb / MakeHuman plugin for Blender**](http://static.makehumancommunity.org/mpfb.html). It maps
the BVH joints onto the MakeHuman armature with copy-rotation bone constraints
(body / hands / feet / face are independently selectable), and exposes a
**"MocapNET BVH Animation Helper"** panel in the 3D viewport's N-panel.

End-to-end workflow:

```bash
# 1) Generate one BVH per detected person
./fast_sam_3dbody_run --from clip.mp4 --bvh ./p.bvh --headless
# → p_0.bvh, p_1.bvh, …

# 2) (One-time) install a known-good Blender + open the plugin
cd blender && ./downloadAndInstallBlender.sh
# downloads blender-3.4.1, launches it with the plugin loaded;
# on first run the plugin will offer to fetch + install the mpfb2 MakeHuman addon
```

Inside Blender:

1. Use the **MakeHuman** (mpfb) tab to create or load a skinned character.
2. `File → Import → Motion Capture (.bvh)` and pick one of the `p_<id>.bvh` files.
   The BVH skeleton appears as a separate armature.
3. Open the **MocapNET BVH Animation Helper** panel, point *Source BVH* at the
   imported armature, tick the body parts you want driven (body / hands /
   feet / face) and click **Apply** — the plugin adds copy-rotation constraints
   on every matching MakeHuman bone.
4. Scrub or render the timeline as usual; the character now follows the
   exported motion.

The plugin auto-handles naming variants between the BVH skeleton (`body.bvh`,
which is what we export against) and the MakeHuman armature, so the
`p_<id>.bvh` files coming out of `--bvh` work without any further renaming.

---

## Offline multi-pass BVH extraction (`offline_sam_3dbody_render`)

`fast_sam_3dbody_run` and `fast_sam_3dbody_render` are **online / causal**
binaries: they process frame N before frame N+1 has been decoded, so every
filter and every tracker can only look backwards.  That works fine for live
webcam input, but for *video files on disk* it leaves quality on the table.

`offline_sam_3dbody_render` is a separate executable that reads the whole
clip first and then runs five passes over it.  It produces the same
multi-person BVH files as `--bvh` does in the live binaries, but with
smoother motion and more stable identities.

```bash
# Most common invocation
./scripts/offline_video.sh --from matrix.mp4 --bvh ./mtx.bvh \
                           --interpolate-jitter

# → mtx_0.bvh, mtx_1.bvh, …, one per tracked identity
```

### What each pass does

1. **Inference + scene detection** — decode the full video, run YOLO +
   backbone + decoder + MHR per frame, throw away `pred_vertices` immediately
   to keep memory bounded.  In parallel we run a scene-change detector
   that votes on three signals each frame (any two ⇒ cut):

   - **(A)** Lucas-Kanade optical-flow tracking of background-only corners
     (41×41 window, 5 pyramid levels — sized for action footage) from the
     previous frame.  Cut signal fires when the LK success rate drops
     below `--scene-success-threshold` (default 0.50).  The "background"
     is everything outside the dilated YOLO bboxes — same mask the user
     asked for, just produced from bbox geometry instead of a GL shader.
   - **(B)** HSV-histogram correlation of the same background.  Cut signal
     fires when the correlation drops below 0.50.
   - **(C)** Person-set discontinuity.  Cut signal fires when either the
     detection count changes by ≥ 2 (or doubles/halves), or the median
     nearest-prev-detection 3D distance exceeds 1 m.

   The voting fuses the failure modes of each individual heuristic, and a
   4-frame debounce throws out tracker-artefact bursts.  Scene cuts then
   gate Pass 3 and Pass 4.

2. **Global identity tracking** — greedy per-frame matching by
   `cost = (1 − IoU) + λ · ‖Δpred_cam_t‖_metres`, then a post-hoc merge
   step that splices track-ending pairs (`A` ends, `B` starts within
   `--track-merge-frames` and `--track-merge-cm` of the same 3D location)
   under `A`'s id.  This is what catches "person hidden behind a pillar
   for 30 frames" — the live tracker would have retired and re-acquired
   them as two separate IDs.  Tracks with fewer than `--min-track-frames`
   detections are pruned at the end (default 8 — drops YOLO single-frame
   false positives).

3. **Gap interpolation** *(on by default; opt out with `--no-gap-interpolation`)*
   — fill in any frame inside a track's lifespan that's missing a
   detection by linear / SLERP-interpolating from the bracketing real
   detections.  Without this, `BVHWriter` pads missing frames with
   "duplicate last pose" and Blender sees that as a frozen segment for
   the whole gap (typical cause: YOLO confidence dipped for a partial
   occlusion, fast head turn, or low-contrast frame).  Scene cuts are
   respected — we never bridge a cut.  `--gap-max-frames N` lets you
   cap how long a gap will be filled; longer gaps revert to padding
   (useful when the two endpoints are too different and the linear
   morph looks unphysical).

4. **Jitter interpolation** *(opt-in via `--interpolate-jitter`)* — flag
   frames whose 3D keypoint velocity exceeds `--jitter-threshold-cm`
   (default 30 cm/frame), then replace each flagged frame by a linear /
   SLERP interpolation of its non-flagged neighbours.  Scene cuts are
   respected — a "200 cm/frame velocity" right at a cut is real motion
   between shots, not noise, and we never bridge across a cut.

5. **Zero-phase smoothing** — same Butterworth (for linear channels) and
   QuatLPF (for `global_rot`) as the live binaries, but run as
   `filtfilt`: forward pass, reverse, forward pass, reverse.  The phase
   shift of the two passes cancels exactly, so the smoothed output has
   **zero temporal lag**.  Each track is split at scene cuts and each
   segment is filtered independently — we never blend pose data from
   shot A into shot B.

6. **BVH export** — feed the smoothed + interpolated detections, with
   the globally-determined track IDs, into
   `BVHWriter::write_frame_external`.  The writer is exactly the one
   used by the live binaries; only the source of the IDs differs.

### Reusing code from the live binaries

`fsb::Pipeline`, `BVHWriter`, `ButterWorth`, `QuatLPF`, `euler_zyx_to_quat`,
`fsb::apply_hand_pose` — all unchanged, called directly.  The new code in
`render/offline_sam_3dbody_render.cpp` is **only** the per-pass
orchestration (≈ 700 lines, every function with a top-of-block comment
explaining why it does what it does), the scene detector, and one new
public method on `BVHWriter` (`write_frame_external`) that takes
caller-supplied track IDs instead of running its internal greedy tracker.

### Restrictions

The binary refuses webcam indices, RTSP URLs, and single still images;
those are an online-binary workflow.  `--from` must be a path to a video
file with at least a handful of frames.

### Full option list (offline-only flags)

| Flag | Default | Notes |
|------|---------|-------|
| `--smoothing zero-phase\|forward\|off` | `zero-phase` | Forward+backward `filtfilt` (no lag) vs the live binaries' forward-only filter vs no smoothing |
| `--no-gap-interpolation` | — | Disable Pass 3; missing frames inside a track will be padded with the last-known pose (the pre-fix "frozen segment" behaviour) |
| `--gap-max-frames N` | `0` | Skip gap-fill for gaps longer than N frames (`0` = no limit).  Useful when long occlusions produce a visibly unphysical morph between two unrelated poses |
| `--interpolate-jitter` | off | Enable Pass 4 |
| `--jitter-threshold-cm CM` | `30.0` | Per-frame 3D-keypoint velocity above which a frame is replaced by an interpolation of its neighbours |
| `--track-merge-frames N` | `30` | Maximum gap (in frames) for the post-hoc merge in Pass 2 |
| `--track-merge-cm CM` | `50.0` | Maximum 3D root distance (cm) for the post-hoc merge |
| `--min-track-frames N` | `8` | Drop tracks with fewer than this many detections (YOLO false-positive filter) |
| `--no-scene-detection` | — | Treat the whole clip as one continuous shot (skips Pass-1 OpenCV work and prevents any false-positive cuts from interrupting the smoothing) |
| `--static-scene` | — | Intent-named alias of `--no-scene-detection`; use when you know the source has no cuts (one-take dance clip, lab recording, …) |
| `--scene-success-threshold T` | `0.50` | LK-flow success-rate threshold (signal A) |
| `--scene-min-corners N` | `30` | Re-seed bg corners when fewer than this many remain |

All `--bvh-*` flags and `--rot-clamp`, `--bw-cutoff` documented above
work identically here.

### Wrapper script

`scripts/offline_video.sh` is the shell entry point.  It mirrors the
`scripts/video.sh` interface (same model paths, same flag forwarding)
and additionally accepts the offline-only flags above.  Usage:

```bash
./scripts/offline_video.sh --from clip.mp4 --bvh out.bvh [...]
```

Passing `--save [vis.mp4]` *additionally* runs `scripts/video.sh` after
the offline pass to produce a visualisation mp4 of the per-frame
inference.  The two outputs are independent — the BVH reflects the
offline-only tracking + smoothing + jitter handling, while the rendered
mp4 is whatever the live renderer would produce on the same clip — so
this is a quick way to get a "what was the input?" video alongside the
"what was the cleaned-up output?" BVH:

```bash
./scripts/offline_video.sh --from matrix.mp4 --bvh mtx.bvh --save vis.mp4
```

---

## C++ library API

```cpp
#include "fast_sam_3dbody.h"

fsb::PipelineConfig cfg;
cfg.onnx_dir        = "./onnx";
cfg.gguf_path       = "./onnx/pipeline.gguf";
cfg.yolo_path       = "./onnx/yolo.onnx";
cfg.cuda_device     = 0;       // -1 = CPU only
cfg.skip_body_model = true;    // faster: no vertices
cfg.max_persons     = 4;       // 0 = unlimited

fsb::Pipeline pipeline;
pipeline.load(cfg);

// BGR uint8 pointer, width, height
std::vector<fsb::MHRResult> results =
    pipeline.process_bgr(bgr_ptr, width, height);

for (const auto& r : results) {
    // r.bbox          [4]   x1 y1 x2 y2 (original image pixels)
    // r.global_rot    [3]   Euler ZYX global orientation
    // r.body_pose     [133] joint angles
    // r.shape         [45]  identity betas
    // r.pred_cam_t    [3]   raw camera head: [s, tx, ty]
    // r.focal_length        estimated focal length (pixels)
    // r.keypoints_yolo[51]  COCO 17 × [x,y,conf]
    // r.pred_vertices [18439*3]  (empty when skip_body_model=true)
}

pipeline.free();
```

## Plain C / ctypes API

```c
#include "fast_sam_3dbody_capi.h"

FsbHandle h = fsb_create();

FsbConfig cfg = {
    .onnx_dir        = "./onnx",
    .gguf_path       = "./onnx/pipeline.gguf",
    .yolo_path       = "./onnx/yolo.onnx",
    .cuda_device     = 0,
    .skip_body_model = 1,
    .person_thresh   = 0.5f,
    .person_nms_iou  = 0.45f,
    .max_persons     = 0,
};
fsb_load(h, &cfg);

FsbResult results[32];
int n = fsb_process_bgr(h, bgr, width, height, results, 32);

for (int i = 0; i < n; i++) {
    // results[i].bbox, .body_pose, .yolo_kps, ...
}

fsb_destroy(h);
```

---

## Performance notes

| Stage | Time (RTX 3090, B=1) |
|-------|----------------------|
| YOLO detection | ~5 ms |
| Backbone (DINOv3-ViT-H) | ~150–200 ms |
| Decoder (6-layer) | ~20 ms |
| MHR + camera FFN (CPU) | <1 ms |
| Native C LBS (optional) | <1 ms |

- Backbone is the bottleneck; it dominates end-to-end latency.
- Use `--skip-body` unless 3D vertices are required.
- For higher throughput, batch multiple crops in a single backbone forward pass (already done when multiple persons are detected).



---

The official PyTorch SAM 3D Body repository from Meta Superintelligence Labs is :

https://github.com/facebookresearch/sam-3d-body

---

## Citation

If you use this software repository in your research or work, please cite:

```bibtex
@misc{qammaz2026sam3dbodycpp,
  author       = {Qammaz, Ammar},
  title        = {{SAM3DBody-cpp}: Standalone {C++} Inference Engine for {SAM-3D-Body}},
  year         = {2026},
  howpublished = {\url{https://github.com/AmmarkoV/SAM3DBody-cpp}},
  note         = {Zero-dependency runtime: ONNX Runtime + ggml, with BVH export and Python ctypes frontends}
}
```

as well as the Fast SAM 3D Body work which was the inspiration creating the cpp version


```bibtex
@article{yang2026fastsam3dbody,
title={Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery},
author={Yang, Timing and He, Sicheng and Jing, Hongyi and Yang, Jiawei and Liu, Zhijian and Zou, Chuhang and Wang, Yue},
journal={arXiv preprint arXiv:2603.15603},
year={2026}
}
```


and of course the Meta AI team behind the awesome paper that proposes the SAM 3D Body method.

```bibtex
@article{yang2026sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint arXiv:2602.15989},
  year={2026}
}
```

