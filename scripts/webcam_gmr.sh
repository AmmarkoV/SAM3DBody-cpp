#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  webcam_gmr.sh
#
#  LIVE webcam ──► humanoid robot, the streaming counterpart of video_gmr.sh.
#
#       webcam ─► fast_sam_3dbody_run --bvh-stream -   (one BVH line / frame)
#              ─► tools/gmr_stream.py                  (per-frame GMR retarget)
#              ─► sink: live MuJoCo viewer (default) | Unitree DDS (stub)
#
#  Unlike video_gmr.sh (offline: whole clip -> multi-pass BVH -> batch retarget),
#  this is causal and never touches disk: the live binary streams one LAFAN BVH
#  MOTION line per frame, gmr_stream.py reassembles + retargets each one as it
#  arrives.  See GMR.md ("Live webcam -> robot") for the design.
#
#  Usage:
#       scripts/webcam_gmr.sh [SOURCE] [RobotType] [-- extra binary flags]
#
#       SOURCE     webcam index / /dev/videoN / a video-file path   (default 0)
#       RobotType  a robot with a LAFAN IK config in GMR            (default unitree_g1)
#
#  Examples:
#       scripts/webcam_gmr.sh                       # webcam 0 -> unitree_g1 viewer
#       scripts/webcam_gmr.sh 2 unitree_g1          # webcam 2
#       scripts/webcam_gmr.sh clean_sample.mp4      # a file, as if it were live
#       SINK=dds scripts/webcam_gmr.sh 0 unitree_g1 # DDS sink (stub; see gmr_stream.py)
#       HEADLESS=1 scripts/webcam_gmr.sh 0          # no input overlay window (robot only)
#
#  Two windows open by default: the input RGB frame with the 2D skeleton overlaid
#  (the binary's own live view), plus the retargeted robot (the sink's viewer).
#  HEADLESS=1 suppresses the input overlay (e.g. on a server / under xvfb).
#
#  Prerequisites:
#    - the C++ binary is built (build/fast_sam_3dbody_run)
#    - tools/setup_gmr.sh has been run (GMR/venv + lafan_mhr.bvh)
#    - a working inference backend (CUDA GPU; the BF16 backbone/decoder need one)
#    - an X display for the MuJoCo viewer (wrap with xvfb-run if headless)
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$THISDIR/.." && pwd )"
cd "$REPO"

# ── args ─────────────────────────────────────────────────────────────────────
SOURCE="${1:-0}"
ROBOT="${2:-unitree_g1}"
EXTRA=()
if [ "${3:-}" = "--" ]; then shift 3 || true; EXTRA=("$@"); fi
SINK="${SINK:-viewer}"

# Show the input RGB frame with the 2D skeleton overlaid (the binary's own live
# window) alongside the robot viewer.  Set HEADLESS=1 to suppress it (servers /
# xvfb, or when you only want the robot).
HEADLESS="${HEADLESS:-0}"
HEADLESS_ARG=()
[ "$HEADLESS" != "0" ] && HEADLESS_ARG=(--headless)

# ── env: project venv + TensorRT libs on LD_LIBRARY_PATH (both no-op if unset) ─
# shellcheck source=/dev/null
source "$REPO/tools/project_env.sh" 2>/dev/null || true
# shellcheck source=/dev/null
source "$REPO/tools/trt_env.sh"     2>/dev/null || true

BIN="$REPO/build/fast_sam_3dbody_run"
GMR_PY="$REPO/GMR/venv/bin/python"
TEMPLATE="$REPO/lafan_mhr.bvh"
POS_CONFIG="$REPO/scripts/gmr_configs/bvh_lafan1pos_to_g1.json"

# ── TensorRT fast path (auto-detected, graceful fallback) ─────────────────────
# The webcam is backbone-bound; the TRT EP (fp16) is ~1.6x faster than the CUDA
# EP.  We turn it on only when EVERYTHING it needs is present, so a box without
# the TRT stack silently keeps working on the default backend:
#   - trt_env.sh resolved the bundled TRT libs onto LD_LIBRARY_PATH (SAM3D_TRT_LIBS)
#   - the TRT-ready models are on disk (backbone_fp16_trt.onnx + decoder_fp16.onnx)
#   - a CUDA GPU exists (nvidia-smi)
# Override with TRT=0 to force the fallback, TRT=1 to force --trt regardless.
# --cuda 0 is added only if the caller didn't already pass --cuda in EXTRA.
TRT="${TRT:-auto}"
TRT_ARG=()
_trt_reason=""
if [ "$TRT" = "0" ] || [ "$TRT" = "off" ]; then
    _trt_reason="disabled (TRT=$TRT)"
elif [ -z "${SAM3D_TRT_LIBS:-}" ] && [ "$TRT" != "1" ] && [ "$TRT" != "on" ]; then
    _trt_reason="TRT libs not found (run tools/setup_trt.sh)"
elif [ ! -e "$REPO/onnx/backbone_fp16_trt.onnx" ] || [ ! -e "$REPO/onnx/decoder_fp16.onnx" ]; then
    _trt_reason="TRT-ready models missing in onnx/ (backbone_fp16_trt.onnx / decoder_fp16.onnx)"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    _trt_reason="no CUDA GPU (nvidia-smi not found)"
else
    TRT_ARG=(--trt)
    case " ${EXTRA[*]-} " in *" --cuda "*) : ;; *) TRT_ARG+=(--cuda 0) ;; esac
fi

for f in "$BIN" "$GMR_PY" "$TEMPLATE" "$POS_CONFIG"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing $f" >&2
        [ "$f" = "$GMR_PY" ]   && echo "       run tools/setup_gmr.sh first." >&2
        [ "$f" = "$TEMPLATE" ] && echo "       run tools/gen_lafan_bvh.py (or tools/setup_gmr.sh)." >&2
        exit 1
    fi
done

# ── transport: shared memory (zero-copy) vs stdout pipe (fallback) ───────────
# The fast path hands each frame's BVH channels to gmr_stream.py over POSIX
# shared memory (SharedMemoryVideoBuffers) instead of an ASCII "@F" stdout line
# — and the driver decodes frames without a per-frame temp .bvh either way.
# We use it only when it's actually available: Linux, the .so built by
# tools/setup_gmr.sh, and a binary compiled with FSB_SHM (it advertises --bvh-shm
# in --help).  Otherwise we transparently fall back to the pipe.  SHM=0 forces
# the pipe; SHM=1 skips the auto-checks.
SHM="${SHM:-auto}"
SHM_LIB="$REPO/SharedMemoryVideoBuffers/libSharedMemoryVideoBuffers.so"
SHM_ACTIVE=0
SHM_DESC=""
_shm_reason=""
if [ "$SHM" = "0" ] || [ "$SHM" = "off" ]; then
    _shm_reason="disabled (SHM=$SHM)"
elif [ "$(uname -s)" != "Linux" ]; then
    _shm_reason="not Linux"
elif [ ! -f "$SHM_LIB" ]; then
    _shm_reason="shm lib missing (run tools/setup_gmr.sh)"
elif ! "$BIN" --help 2>&1 | grep -q -- "--bvh-shm"; then
    _shm_reason="binary built without FSB_SHM (rebuild after tools/setup_gmr.sh)"
else
    SHM_ACTIVE=1
    SHM_DESC="sam3dbody_bvh_$$.shm"   # per-PID name so parallel runs don't collide
fi

echo "[webcam_gmr] source=$SOURCE robot=$ROBOT sink=$SINK headless=$HEADLESS"
if [ "${#TRT_ARG[@]}" -gt 0 ]; then
    echo "[webcam_gmr] TensorRT fast path ON (${TRT_ARG[*]})"
else
    echo "[webcam_gmr] TensorRT off -> default backend  [$_trt_reason]"
fi
if [ "$SHM_ACTIVE" = 1 ]; then
    echo "[webcam_gmr] transport: shared memory ($SHM_DESC)  (Ctrl-C to stop)"
else
    echo "[webcam_gmr] transport: stdout pipe  [$_shm_reason]  (Ctrl-C to stop)"
fi
[ "$HEADLESS" = "0" ] && echo "[webcam_gmr] input overlay window on (press q in it to stop; HEADLESS=1 to disable)"

# --max-persons 1: one actor drives the robot. --butterworth (+ root rotation):
# the live binary's causal smoothing, so the robot isn't fed raw per-frame jitter.
# --flip-depth: MHR is camera-space; its depth sign is opposite to GMR's (robot
# would moonwalk otherwise) — same rationale as video_gmr.sh.
BIN_ARGS=(
    --onnx-dir "$REPO/onnx"
    --gguf     "$REPO/onnx/pipeline.gguf"
    --yolo     "$REPO/onnx/yolo.onnx"
    --from     "$SOURCE"
    --max-persons 1
    --bvh-template "$TEMPLATE"
    --butterworth --butterworth-root-rotation
    "${TRT_ARG[@]}"
    "${HEADLESS_ARG[@]}"
    "${EXTRA[@]}"
)
PY_ARGS=(
    --robot    "$ROBOT"
    --config   "$POS_CONFIG"
    --template "$TEMPLATE"
    --sink     "$SINK"
    --flip-depth
)

if [ "$SHM_ACTIVE" = 1 ]; then
    # Two independent processes sharing a POSIX shm segment (no stdin pipe): the
    # binary publishes frames; the driver reads them.  Clean up the descriptor
    # and the publisher on exit.
    cleanup_shm() {
        [ -n "${BIN_PID:-}" ] && kill "$BIN_PID" 2>/dev/null || true
        rm -f "/dev/shm/$SHM_DESC" 2>/dev/null || true
    }
    trap cleanup_shm EXIT INT TERM
    "$BIN" "${BIN_ARGS[@]}" --bvh-shm "$SHM_DESC" &
    BIN_PID=$!
    # ShmBvhReader waits for the descriptor + feed, so the driver may start first.
    "$GMR_PY" "$REPO/tools/gmr_stream.py" "${PY_ARGS[@]}" \
        --bvh-shm "$SHM_DESC" --shm-stream bvh --shm-lib "$SHM_LIB"
else
    # Fallback: "@F <channels>" per frame on the binary's stdout, piped in.
    "$BIN" "${BIN_ARGS[@]}" --bvh-stream - \
      | "$GMR_PY" "$REPO/tools/gmr_stream.py" "${PY_ARGS[@]}"
fi
