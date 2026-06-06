# Dependencies & Runtime Troubleshooting

This document covers the runtime dependencies of the C++ pipeline and how to
diagnose the most common startup failures. If the project **builds** but **fails
at runtime**, you are almost certainly in the right place.

---

## 1. Runtime dependency matrix

| Component | Version | Notes |
|-----------|---------|-------|
| ONNX Runtime | **1.20.1 (GPU build)** | Downloaded automatically by CMake into `build/onnxruntime_dl/` if not found. |
| CUDA | **12.x** | Required by the ORT 1.20.1 CUDA execution provider. |
| cuDNN | **9.x** | Required by ORT ≥ 1.19. **cuDNN 8 will not work** and is the #1 cause of the CUDA EP failing to load. |
| NVIDIA driver | Recent enough for CUDA 12 (≥ 525) | Check with `nvidia-smi`. |
| TensorRT | **10.4** *(optional)* | Only needed for the `--trt` execution provider (~1.6× faster). ORT 1.20.1 dlopens `libnvinfer.so.10`; install into `tools/.venv` and run via `tools/run_trt.sh`. Without it, `--trt` falls back to the CUDA EP — see §6. |
| OpenCV | core / imgproc / videoio / highgui / dnn | System package. |

> The standard `onnx/backbone.onnx` is exported in **BFloat16** and runs **only on
> the CUDA execution provider**. There is no CPU BF16 path — see §3.
>
> On CUDA you can optionally run a **float16** backbone (`backbone_fp16.onnx`,
> produced by `tools/export_backbone_fp16.py`): ~3× smaller on disk/VRAM and a few
> percent faster, with identical output. When present in `onnx/`, the loader
> prefers it automatically; pin `--backbone backbone.onnx` to force the bf16 model.

---

## 2. Symptom: `Failed to load library libonnxruntime_providers_cuda.so`

This means the CUDA execution provider shared library could not be loaded, so
ONNX Runtime silently falls back to CPU (which then fails — see §3). The provider
`.so` itself is present; one of its **dependencies** is missing or the wrong
version — virtually always **cuDNN 9** or a CUDA 12 runtime lib.

### Diagnose

```bash
# 1. Driver present, and which CUDA version it supports?
nvidia-smi

# 2. THE decisive command — what is the provider .so actually missing?
ldd build/onnxruntime_dl/lib/libonnxruntime_providers_cuda.so | grep -i "not found"

# 3. Is cuDNN 9 / cuBLAS installed and visible to the loader?
ldconfig -p | grep -i "cudnn\|cublas\|cufft"
```

Step 2 prints the exact missing libraries — typically `libcudnn.so.9`,
`libcublasLt.so.12`, or `libcublas.so.12`.

### Fix A — system packages (Ubuntu 22.04, needs root)

```bash
sudo apt-get install -y cuda-toolkit-12-4 libcudnn9-cuda-12
sudo ldconfig
```

### Fix B — pip wheels (no root; install into the venv you run from)

```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Make the loader see the pip-installed libs before running:
export LD_LIBRARY_PATH=$(python -c "import os, nvidia.cudnn, nvidia.cublas; \
print(':'.join(os.path.join(os.path.dirname(m.__file__), 'lib') \
for m in (nvidia.cudnn, nvidia.cublas)))"):$LD_LIBRARY_PATH
```

### Verify

```bash
# Should now print nothing (no missing libraries):
ldd build/onnxruntime_dl/lib/libonnxruntime_providers_cuda.so | grep -i "not found"
```

Re-run `scripts/webcam.sh` — the CUDA EP should load and the pipeline should run
on the GPU.

---

## 3. Symptom: `Could not find an implementation for Expand(13)` (on CPU)

**This is expected, and it is not a CPU bug to fix.** When the CUDA EP fails to
load (§2), ORT falls back to the CPU provider. The standard `backbone.onnx` is
**BFloat16**, and the ORT CPU provider has **no BF16 kernel for `Expand`** (and
other ops), so it dies on the first BF16 operator it hits.

The correct response is to **fix the CUDA provider (§2)**, not to chase the
`Expand` error. There are only two ways to actually run the pipeline:

1. **Get the CUDA EP loading** (recommended — see §2), then use the standard
   `backbone.onnx`.
2. **Run on CPU with the float32 backbone** — a separate download that has no
   BF16 ops. This is slow (≈ 5–15 s per backbone pass; video is impractical):

   ```bash
   # Download the fp32 backbone alongside the other models in onnx/
   wget -P onnx/ https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone_fp32.onnx
   wget -P onnx/ https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/backbone_fp32.onnx.data

   ./build/fast_sam_3dbody_run \
       --onnx-dir ./onnx \
       --backbone backbone_fp32.onnx \
       --cuda     -1 \
       --from     your_image.png
   ```

---

## 4. WSL2 notes

If `nvidia-smi` works inside WSL, install the
[CUDA toolkit for WSL2](https://developer.nvidia.com/cuda-downloads) (the **WSL2
toolkit only** — do *not* install the full Linux driver inside WSL) and use the
standard `backbone.onnx`.

---

## 6. TensorRT execution provider (`--trt`, optional)

The TensorRT EP is **compiled in by default** for CUDA builds (CMake option
`SAM3D_TENSORRT=ON`) but only used at runtime when you pass `--trt`. It builds a
fused, FP16 TensorRT engine per ONNX graph, which can beat the plain CUDA EP on
the heavy backbone/decoder.

### Requirements

ORT 1.20.1 links the TensorRT EP against **TensorRT 10.4** — it dlopens
`libnvinfer.so.10`, `libnvonnxparser.so.10` and `libnvinfer_plugin.so.10` at
session-create time. These are **not** bundled. Install the matching wheels into
`tools/.venv` and run through the `tools/run_trt.sh` wrapper, which puts them on
`LD_LIBRARY_PATH` and adds `--trt --cuda 0` for you:

```bash
python3 -m venv tools/.venv
tools/.venv/bin/pip install "tensorrt-cu12-libs==10.4.0" --extra-index-url https://pypi.nvidia.com

tools/run_trt.sh --onnx-dir ./onnx --from your_video.mp4
```

> Pin **10.4** — it is the TensorRT version ORT 1.20.1 was built against. The
> plain `tensorrt-cu12` metapackage may not pull the `…-libs` payload on its own,
> so install `tensorrt-cu12-libs` explicitly as above.

The launcher scripts (`scripts/webcam.sh`, `webcam_p.sh`, `video.sh`,
`offline_video.sh`) source `tools/trt_env.sh` automatically, so `--trt` just works
there. To wire it up manually in your own shell, source the same helper (it
auto-detects the venv's `python<ver>` dir):

```bash
source tools/trt_env.sh
./build/fast_sam_3dbody_run --trt --cuda 0 --onnx-dir ./onnx --from your_video.mp4
```

A system `.deb`/`.tar` TensorRT 10.4 install from developer.nvidia.com/tensorrt
(then `sudo ldconfig`) works too. Verify the runtime is visible:

```bash
ldd build/onnxruntime_dl/lib/libonnxruntime_providers_tensorrt.so | grep -i nvinfer
# every libnvinfer*.so.10 / libnvonnxparser.so.10 line should resolve (no "not found")
```

### Behaviour without TensorRT installed

`--trt` degrades gracefully. The loader tries **TensorRT → CUDA → CPU** per
session, so a missing runtime just logs and falls back to the CUDA EP:

```
[ORT] TensorRT EP failed for './onnx/backbone.onnx' (… libnvinfer.so.10:
      cannot open shared object file …)
[ORT] Falling back to CUDA…
```

The pipeline still runs on the GPU — you simply don't get the TRT speedup.

### Engine cache

The first `--trt` run **builds** the TensorRT engines, which can take several
minutes, and writes them to `onnx/trt_engine_cache/` (next to the model).
Subsequent runs reuse the cache and start quickly. Caveat: engines are keyed by
input shape, so each distinct person-count batch size triggers its own one-time
build; engines are also invalidated by TensorRT/GPU/driver upgrades (delete the
cache dir to force a rebuild).

To compile the EP out entirely: `cmake -DSAM3D_TENSORRT=OFF ..`.

### Model compatibility — auto-handled under `--trt`

The stock `backbone.onnx` / `decoder.onnx` are **not** TRT-buildable as exported,
so `--trt` automatically swaps in TRT-friendly variants (see
`resolve_backbone_defaults` in `src/cli_common.h`). Two issues, both worked around:

- **backbone** — the DINOv3 backbone has `If` control-flow nodes (`/rope_embed/If_*`)
  with no static output shape; TRT rejects them (`… has no shape specified …`).
  `--trt` uses **`backbone_fp16_trt.onnx`**, which has those `If` branches folded out.
- **decoder** — `decoder.onnx` is bfloat16, and TRT aborts at inference with
  `TensorRT EP output tensor data type: 16 not supported`. `--trt` uses
  **`decoder_fp16.onnx`** (bf16→fp16, regenerate with
  `tools/export_backbone_fp16.py --input onnx/decoder.onnx --output onnx/decoder_fp16.onnx`).

With those in place, `--trt` gives a measured **~1.6×** end-to-end speedup on an
RTX 1000 Ada laptop (backbone 230→148 ms, total 273→170 ms / 3.7→5.9 fps), keypoints
within ~4 mm of the CUDA path. Confirm it engaged via the `[cli] TRT:` log lines —
if you instead see `Falling back to CUDA…`, the TensorRT libs aren't on the loader
path (see Requirements above).

> **INT8 does not help** and is intentionally not wired up: ORT's CUDA EP runs
> `MatMulInteger` by dequantising to fp32 (~17× *slower* than fp16 on this ViT-H),
> and the only real int8-tensor-core route (TensorRT static QDQ) needs >32 GB host
> RAM to calibrate a model this large. Stick with FP16 + TensorRT.

---

## 7. Quick reference

| You see… | Root cause | Do this |
|----------|-----------|---------|
| `Failed to load library libonnxruntime_providers_cuda.so` | cuDNN 9 / CUDA 12 lib missing | §2 — `ldd … \| grep "not found"`, then install cuDNN 9 |
| `Could not find an implementation for Expand(13)` | CPU fallback hit the BF16 backbone | Fix the CUDA EP (§2), or use `backbone_fp32.onnx --cuda -1` (§3) |
| `Failed to load library libonnxruntime_providers_tensorrt.so … libnvinfer.so.10` | `--trt` used but TensorRT 10 not installed | Install TensorRT 10 (§6), or drop `--trt` — it auto-falls back to CUDA |
