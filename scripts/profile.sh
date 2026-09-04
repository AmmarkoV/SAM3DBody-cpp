#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  profile.sh
#
#  Profiles ./build/fast_sam_3dbody_render with `valgrind --tool=callgrind`
#  and opens the result in kcachegrind (falls back to qcachegrind, then to
#  a text report via callgrind_annotate if neither GUI is installed —
#  `sudo apt install kcachegrind`).
#
#  NOTE: callgrind only instruments CPU instructions. GPU/TensorRT kernel
#  time (e.g. when passing --trt) is invisible to it — the profile shows
#  CPU-side cost (pre/post-processing, driver call overhead, BVH/mesh math),
#  not GPU compute. valgrind also slows execution down roughly 20-50x, so
#  this defaults to a bounded run against a local video file with a frame
#  cap rather than the live webcam, which has no natural end point.
#
#  Usage:
#     scripts/profile.sh [--trt] [--max-persons N] [--refined-pose] [...]
#     scripts/profile.sh --from other.mp4 --frames 60 ...
#
#  All args are forwarded to fast_sam_3dbody_render, same convention as
#  webcam.sh / video.sh. These defaults are added ONLY if not already
#  present among the forwarded args:
#     --from     ./multi.mp4
#     --frames   30
#     --headless
# ════════════════════════════════════════════════════════════════════════════

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

ARGS=("$@")
HAS_FROM=0
HAS_FRAMES=0
HAS_HEADLESS=0
for a in "${ARGS[@]}"; do
    [ "$a" = "--from" ]     && HAS_FROM=1
    [ "$a" = "--frames" ]   && HAS_FRAMES=1
    [ "$a" = "--headless" ] && HAS_HEADLESS=1
done

DEFAULTS=()
[ $HAS_FROM     -eq 0 ] && DEFAULTS+=(--from ./multi.mp4)
[ $HAS_FRAMES   -eq 0 ] && DEFAULTS+=(--frames 30)
[ $HAS_HEADLESS -eq 0 ] && DEFAULTS+=(--headless)

OUTDIR=/tmp/sam3dbody_profile
mkdir -p "$OUTDIR"
CGOUT="$OUTDIR/callgrind.out.%p"

echo "── profiling with valgrind --tool=callgrind ───────────────────────────────"
echo "   output: $CGOUT"
echo "   (this runs roughly 20-50x slower than realtime — be patient)"
echo

valgrind --tool=callgrind \
    --callgrind-out-file="$CGOUT" \
    --collect-jumps=yes \
    --collect-systime=yes \
    ./build/fast_sam_3dbody_render \
        --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf --yolo ./onnx/yolo.onnx \
        --mesh ./body_mesh.tri --lbs onnx/body_model.lbs \
        "${DEFAULTS[@]}" "${ARGS[@]}"
RUN_EXIT=$?

CGFILE=$(ls -t "$OUTDIR"/callgrind.out.* 2>/dev/null | head -1)
if [ -z "$CGFILE" ]; then
    echo "no callgrind output file found in $OUTDIR — valgrind may have failed to start." >&2
    exit ${RUN_EXIT:-1}
fi

echo
echo "── profile written: $CGFILE ────────────────────────────────────────────────"

if command -v kcachegrind >/dev/null 2>&1; then
    echo "launching kcachegrind..."
    kcachegrind "$CGFILE" &
elif command -v qcachegrind >/dev/null 2>&1; then
    echo "kcachegrind not found, launching qcachegrind..."
    qcachegrind "$CGFILE" &
elif command -v callgrind_annotate >/dev/null 2>&1; then
    echo "kcachegrind not installed (sudo apt install kcachegrind for the GUI)."
    echo "showing a text report via callgrind_annotate instead:"
    echo
    callgrind_annotate "$CGFILE" | head -100
else
    echo "Neither kcachegrind nor callgrind_annotate found. Install one to inspect the profile:"
    echo "  sudo apt install kcachegrind"
    echo "  callgrind_annotate $CGFILE"
fi

exit $RUN_EXIT
