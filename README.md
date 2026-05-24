# SAM3DBody-cpp

Standalone C++ inference engine for **SAM-3D-Body** — zero Python dependency at runtime.

Takes a BGR image and produces per-person MHR body pose parameters, camera translation, and optionally full 3D mesh vertices + 70 body/hand keypoints, all via ONNX Runtime + ggml.

Also includes Python frontends that call the compiled shared library via ctypes, and a CSV exporter for the 70 MHR keypoints.

![SAM3DBody-cpp](doc/screen.jpg)

---

## Models

Pre-built ONNX / GGUF / LBS model files are hosted on HuggingFace:

**[https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models](https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models)**

Download `SAM3DBody-cpp-onnx-models.zip`, extract it, and place the resulting `onnx/` directory at the root of this repository:

```bash
# Download and extract
wget https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-onnx-models.zip
unzip SAM3DBody-cpp-onnx-models.zip
# onnx/ is now at the repo root — ready to build
```

| File | Size | Description |
|------|------|-------------|
| `onnx/backbone.onnx` + `.data` | ~4.8 GB | DINOv3-ViT-H/14+ encoder |
| `onnx/decoder.onnx` | ~93 MB | 6-layer PromptableDecoder |
| `onnx/yolo.onnx` | ~81 MB | YOLO11m-pose person detector |
| `onnx/pipeline.gguf` | ~5 MB | MHR + camera projection heads |
| `onnx/body_model.lbs` | ~27 MB | Native C LBS data (joints, weights, shape) |
| `onnx/correctives.bin` | ~33 MB | Pose corrective blend shapes |
| `onnx/keypoint_mapping.bin` | ~8 KB | MHR-70 keypoint index map |

> **CMake will warn** at configure time if neither `onnx/` nor the zip is found.

---

## Pipeline

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
│   └── build.sh / setup.sh / webcam.sh / video.sh
└── src/
    ├── fast_sam_3dbody.h             C++ public API
    ├── fast_sam_3dbody.cpp           Pipeline implementation
    ├── fast_sam_3dbody_capi.h        Plain C API (for ctypes)
    ├── fast_sam_3dbody_capi.cpp
    ├── preprocess.hpp                Crop, normalise, ray_cond, NMS, pose conversion
    └── main.cpp                      CLI executable (--out CSV, live overlay window)
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


