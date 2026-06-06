#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Put the bundled TensorRT libs on LD_LIBRARY_PATH so --trt works (no-op otherwise).
source "$THISDIR/../tools/trt_env.sh"

./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 $@ # > /tmp/render_raw.txt 


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
