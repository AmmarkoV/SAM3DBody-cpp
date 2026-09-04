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

./build/fast_sam_3dbody_render --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --mesh ./body_mesh.tri --lbs  onnx/body_model.lbs --from /dev/video0 $@  > /tmp/render_raw.txt 


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
