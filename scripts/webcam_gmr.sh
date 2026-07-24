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

# ── env: project venv + TensorRT libs on LD_LIBRARY_PATH (both no-op if unset) ─
# shellcheck source=/dev/null
source "$REPO/tools/project_env.sh" 2>/dev/null || true
# shellcheck source=/dev/null
source "$REPO/tools/trt_env.sh"     2>/dev/null || true

BIN="$REPO/build/fast_sam_3dbody_run"
GMR_PY="$REPO/GMR/venv/bin/python"
TEMPLATE="$REPO/lafan_mhr.bvh"
POS_CONFIG="$REPO/scripts/gmr_configs/bvh_lafan1pos_to_g1.json"

for f in "$BIN" "$GMR_PY" "$TEMPLATE" "$POS_CONFIG"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing $f" >&2
        [ "$f" = "$GMR_PY" ]   && echo "       run tools/setup_gmr.sh first." >&2
        [ "$f" = "$TEMPLATE" ] && echo "       run tools/gen_lafan_bvh.py (or tools/setup_gmr.sh)." >&2
        exit 1
    fi
done

echo "[webcam_gmr] source=$SOURCE robot=$ROBOT sink=$SINK"
echo "[webcam_gmr] webcam -> BVH stream -> GMR retarget -> $SINK  (Ctrl-C to stop)"

# --max-persons 1: one actor drives the robot. --butterworth (+ root rotation):
# the live binary's causal smoothing, so the robot isn't fed raw per-frame jitter.
# --bvh-stream -: emit "@F <channels>" per frame on stdout, piped into the driver.
# --flip-depth: MHR is camera-space; its depth sign is opposite to GMR's (robot
# would moonwalk otherwise) — same rationale as video_gmr.sh.
"$BIN" \
    --onnx-dir "$REPO/onnx" \
    --gguf     "$REPO/onnx/pipeline.gguf" \
    --yolo     "$REPO/onnx/yolo.onnx" \
    --from     "$SOURCE" \
    --max-persons 1 \
    --bvh-template "$TEMPLATE" \
    --bvh-stream   - \
    --butterworth --butterworth-root-rotation \
    --headless \
    "${EXTRA[@]}" \
  | "$GMR_PY" "$REPO/tools/gmr_stream.py" \
        --robot    "$ROBOT" \
        --config   "$POS_CONFIG" \
        --template "$TEMPLATE" \
        --sink     "$SINK" \
        --flip-depth
