#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Make the TensorRT EP usable when --trt is passed: ORT 1.20.1 dlopens
# libnvinfer.so.10 etc., which we ship inside tools/.venv (see DEPENDENCIES.md §6).
# Prepend that dir to LD_LIBRARY_PATH if present — harmless without --trt.
TRT_LIBS="$PWD/tools/.venv/lib/python3.12/site-packages/tensorrt_libs"
if [ -d "$TRT_LIBS" ]; then
    export LD_LIBRARY_PATH="$TRT_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

./build/fast_sam_3dbody_render --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --mesh ./body_mesh.tri --lbs  onnx/body_model.lbs --from /dev/video0 $@  > /tmp/render_raw.txt 


#./build/fast_sam_3dbody_run --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx --from /dev/video0 > /tmp/render_raw.txt $@ 

exit 0
