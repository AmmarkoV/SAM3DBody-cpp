#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 $@ # > /tmp/render_raw.txt 


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
