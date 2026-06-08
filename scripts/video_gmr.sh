#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  video_gmr.sh
#
#  End-to-end pipeline:
#       video file  ──►  LAFAN BVH (SAM3DBody offline extractor)
#                   ──►  humanoid-robot motion (GMR retargeting)
#
#  For every person detected in the clip it produces, in $REPO/gmr_out/<name>/:
#       <name>_<id>.bvh             LAFAN-skeleton motion capture
#       <name>_<id>_<robot>.pkl     robot motion (root_pos/root_rot/dof_pos)
#       <name>_<id>_<robot>.mp4     rendered video of the retargeted robot
#
#  Usage:
#       scripts/video_gmr.sh <videoFile.mp4> [RobotType]
#
#  RobotType defaults to unitree_g1.  Robots that ship a LAFAN IK config in GMR:
#       unitree_g1   fourier_n1   engineai_pm01   booster_t1   stanford_toddy
#
#  Prerequisites:
#    - tools/setup_gmr.sh has been run (creates $REPO/GMR/venv)
#    - an X display is available: GMR's viewer opens a window even when
#      recording (run under `xvfb-run` if you are headless).
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$THISDIR/.." && pwd )"

# ── args ─────────────────────────────────────────────────────────────────────
VIDEO="${1:-}"
ROBOT="${2:-unitree_g1}"

if [ -z "$VIDEO" ]; then
    echo "Usage: $(basename "$0") <videoFile.mp4> [RobotType]" >&2
    exit 2
fi
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: video not found: $VIDEO" >&2
    exit 2
fi
VIDEO="$(realpath "$VIDEO")"   # absolute: the offline wrapper cd's to repo root

# ── GMR venv ─────────────────────────────────────────────────────────────────
GMR_DIR="$REPO/GMR"
VENV_PY="$GMR_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: GMR venv not found ($VENV_PY)." >&2
    echo "       Run tools/setup_gmr.sh first." >&2
    exit 1
fi

BASE="$(basename "${VIDEO%.*}")"
OUT="$REPO/gmr_out/$BASE"
mkdir -p "$OUT"

# ── 1) video -> LAFAN BVH (one <BASE>_<id>.bvh per tracked person) ───────────
echo "[video_gmr] 1/2  $BASE: video -> LAFAN BVH"
"$REPO/scripts/offline_video.sh" \
    --from "$VIDEO" \
    --bvh  "$OUT/${BASE}.bvh" \
    --bvh-template "$REPO/lafan.bvh" \
    --interpolate-jitter

# ── 2) each person BVH -> robot motion (pkl) + rendered video (mp4) ──────────
echo "[video_gmr] 2/2  BVH -> robot ($ROBOT)"
shopt -s nullglob
bvhs=( "$OUT/${BASE}"_*.bvh )
if [ ${#bvhs[@]} -eq 0 ]; then
    echo "ERROR: no BVH produced — was anyone detected in the clip?" >&2
    exit 1
fi

# We use our own driver (tools/gmr_retarget.py) with a POSITION-based GMR config
# instead of GMR's stock bvh_to_robot.py. GMR's stock LAFAN config is orientation-
# driven and calibrated for the real Ubisoft LAFAN1 +X-bone joint frames, which our
# MHR-derived skeleton does not reproduce; retargeting by joint POSITION (our
# positions are correct) sidesteps that. The driver also grounds the feet each
# frame (our data is camera-space). See GMR.md for the full rationale.
POS_CONFIG="$REPO/scripts/gmr_configs/bvh_lafan1pos_to_g1.json"
for bvh in "${bvhs[@]}"; do
    id="$(basename "$bvh" .bvh)"        # e.g. football_0
    echo "[video_gmr]   retargeting $id -> $ROBOT"
    # Run from GMR_DIR so the package finds its ik_configs/ and assets/.
    ( cd "$GMR_DIR" && "$VENV_PY" "$REPO/tools/gmr_retarget.py" \
        --bvh    "$bvh" \
        --robot  "$ROBOT" \
        --config "$POS_CONFIG" \
        --save   "$OUT/${id}_${ROBOT}.pkl" \
        --video  "$OUT/${id}_${ROBOT}.mp4" )
done

echo "[video_gmr] done. Outputs in: $OUT"
ls -1 "$OUT"
