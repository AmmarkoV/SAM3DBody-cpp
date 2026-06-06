#!/usr/bin/env bash
# tools/run_trt.sh — run fast_sam_3dbody with the ONNX Runtime TensorRT EP.
#
# WHY THIS EXISTS
# ---------------
# ORT 1.20.1's TensorRT EP dynamically loads libnvinfer.so.10 / libnvonnxparser.so.10
# / libnvinfer_plugin.so.10 (TensorRT 10.x).  We don't ship a system TensorRT; the
# matching libs live inside tools/.venv (pip `tensorrt-cu12-libs==10.4.0`, the
# version ORT 1.20.1 was built against).  This wrapper puts that tensorrt_libs dir
# on LD_LIBRARY_PATH and then execs the run binary with --trt, so the TRT EP loads
# instead of silently falling back to the CUDA EP.
#
# SETUP (one-time):
#   python3 -m venv tools/.venv
#   tools/.venv/bin/pip install "tensorrt-cu12-libs==10.4.0" --extra-index-url https://pypi.nvidia.com
#
# USAGE (forwards all args to the binary; --trt --cuda 0 are added if absent):
#   tools/run_trt.sh --onnx-dir ./onnx --from ./Buffalo.webm --frames 5 --headless
#   tools/run_trt.sh --from input.mp4 --bvh out.bvh
#
# The first run per model+shape builds the TensorRT engine (minutes) and caches it
# in <onnx-dir>/trt_engine_cache; later runs reuse it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRT_LIBS="$ROOT/tools/.venv/lib/python3.12/site-packages/tensorrt_libs"
BIN="${SAM3D_BIN:-$ROOT/build/fast_sam_3dbody_run}"

if [ ! -d "$TRT_LIBS" ]; then
    echo "tools/run_trt.sh: TensorRT libs not found at $TRT_LIBS" >&2
    echo "  Install them with:" >&2
    echo "    $ROOT/tools/.venv/bin/pip install 'tensorrt-cu12-libs==10.4.0' --extra-index-url https://pypi.nvidia.com" >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "tools/run_trt.sh: binary not found/executable: $BIN  (build it, or set SAM3D_BIN)" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$TRT_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Add --trt / --cuda 0 only if the caller didn't pass them.
args=("$@")
case " $* " in *" --trt "*) : ;; *) args=(--trt "${args[@]}") ;; esac
case " $* " in *" --cuda "*) : ;; *) args=(--cuda 0 "${args[@]}") ;; esac

exec "$BIN" "${args[@]}"
