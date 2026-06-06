# tools/trt_env.sh — sourceable helper (do NOT exec; it must export into your shell)
#
# Puts the bundled TensorRT runtime libs on LD_LIBRARY_PATH so ORT 1.20.1's
# TensorRT EP can dlopen libnvinfer.so.10 / libnvonnxparser.so.10 /
# libnvinfer_plugin.so.10 (see DEPENDENCIES.md §6).  We ship them inside
# tools/.venv (pip `tensorrt-cu12-libs==10.4.0`).  No-op if not installed, so
# sourcing it is always safe even when you aren't passing --trt.
#
# Usage (from any wrapper script):
#   source "$(dirname "${BASH_SOURCE[0]}")/../tools/trt_env.sh"
#
# Auto-detects the venv's python<ver> directory via a glob, so it is not pinned
# to one Python version.  Exports SAM3D_TRT_LIBS = the resolved dir ("" if none).

# Resolve repo root from THIS file's location (independent of the caller's CWD).
_trt_root="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )/.." && pwd )"

SAM3D_TRT_LIBS=""
for _trt_d in "$_trt_root"/tools/.venv/lib/python*/site-packages/tensorrt_libs; do
    [ -d "$_trt_d" ] || continue          # glob didn't match → skip the literal pattern
    SAM3D_TRT_LIBS="$_trt_d"
    export LD_LIBRARY_PATH="$_trt_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
done
export SAM3D_TRT_LIBS
unset _trt_root _trt_d
