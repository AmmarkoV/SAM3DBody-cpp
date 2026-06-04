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
| OpenCV | core / imgproc / videoio / highgui / dnn | System package. |

> The standard `onnx/backbone.onnx` is exported in **BFloat16** and runs **only on
> the CUDA execution provider**. There is no CPU BF16 path — see §3.

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

## 5. Quick reference

| You see… | Root cause | Do this |
|----------|-----------|---------|
| `Failed to load library libonnxruntime_providers_cuda.so` | cuDNN 9 / CUDA 12 lib missing | §2 — `ldd … \| grep "not found"`, then install cuDNN 9 |
| `Could not find an implementation for Expand(13)` | CPU fallback hit the BF16 backbone | Fix the CUDA EP (§2), or use `backbone_fp32.onnx --cuda -1` (§3) |
