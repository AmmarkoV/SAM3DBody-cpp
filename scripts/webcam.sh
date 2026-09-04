#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

# All args are forwarded as-is to fast_sam_3dbody_render, e.g.:
#   --color R G B   tint every mesh in the 3D overlay with one RGB color
#                   (0-255 each), e.g. --color 255 255 0 for yellow.
#
# --nvidia-laptop  Consumed by this script, not forwarded.  For Optimus/PRIME
#                   hybrid-graphics laptops where the X server's default GLX
#                   vendor is the integrated GPU: forces this run onto the
#                   NVIDIA discrete GPU via PRIME render offload, so the CUDA/
#                   TensorRT inference path actually gets the dGPU instead of
#                   silently falling back to whatever the default screen has.
#                   No-op (and harmless) on a desktop/single-GPU or
#                   dGPU-only-mode machine.
ARGS=()
NVIDIA_LAPTOP=0
for a in "$@"; do
    if [ "$a" = "--nvidia-laptop" ]; then
        NVIDIA_LAPTOP=1
    else
        ARGS+=("$a")
    fi
done

if [ "$NVIDIA_LAPTOP" = "1" ]; then
    echo "[webcam.sh] --nvidia-laptop: forcing PRIME render offload onto the NVIDIA GPU"
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

./build/fast_sam_3dbody_render --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --mesh ./body_mesh.tri --lbs  onnx/body_model.lbs --from /dev/video0 "${ARGS[@]}"  > /tmp/render_raw.txt


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
