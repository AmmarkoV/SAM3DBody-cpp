#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  setup_gmr.sh
#
#  Sets up the GMR (General Motion Retargeting) Python environment used by
#  scripts/video_gmr.sh to turn the LAFAN BVH we export into humanoid-robot
#  motion.
#
#  GMR itself lives at  $REPO/GMR  (a symlink to your GMR checkout) — this
#  script does NOT clone it; it only creates a venv inside it and installs the
#  runtime deps:
#    - PyTorch with CUDA (the default Linux PyPI wheel is the CUDA build)
#    - everything GMR needs EXCEPT `smplx`, which is only used by the SMPL-X
#      input path; the BVH/LAFAN path we drive does not need it (so we also
#      skip its ~PyTorch-dragging git build and the ~1 GB SMPL-X body models).
#
#  Idempotent: re-running it reuses the existing venv and just re-checks deps.
#
#  Usage:   tools/setup_gmr.sh
#  Override the interpreter with:   PYTHON=/path/to/python3.10+  tools/setup_gmr.sh
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$THISDIR/.." && pwd )"
GMR_DIR="$REPO/GMR"
VENV="$GMR_DIR/venv"

# ── GMR must already be present (symlink or checkout) ────────────────────────
if [ ! -f "$GMR_DIR/setup.py" ]; then
    echo "ERROR: GMR not found at $GMR_DIR (expected a GMR checkout or symlink)." >&2
    echo "       Create it first, e.g.:" >&2
    echo "           ln -s /path/to/GMR \"$GMR_DIR\"" >&2
    exit 1
fi

# ── Need Python >= 3.10 ──────────────────────────────────────────────────────
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
    echo "ERROR: need Python >= 3.10 (got: $("$PY" --version 2>&1))." >&2
    echo "       Set PYTHON=/path/to/python3.10+ and re-run." >&2
    exit 1
fi

echo "[setup_gmr] GMR dir : $GMR_DIR"
echo "[setup_gmr] python  : $("$PY" --version 2>&1)"

# ── venv ─────────────────────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[setup_gmr] creating venv at $VENV"
    "$PY" -m venv "$VENV"
else
    echo "[setup_gmr] reusing existing venv at $VENV"
fi

PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip

# Install GMR itself WITHOUT its declared deps (so the smplx git dep is skipped)
echo "[setup_gmr] installing GMR package (no deps)"
"$PIP" install -e "$GMR_DIR" --no-deps

# ...then the runtime deps minus smplx, plus a CUDA build of torch.
# (Set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu for a CPU-only torch.)
echo "[setup_gmr] installing runtime deps + torch"
TORCH_ARGS=()
[ -n "${TORCH_INDEX_URL:-}" ] && TORCH_ARGS=(--index-url "$TORCH_INDEX_URL")
"$PIP" install \
    loop_rate_limiters mink mujoco numpy scipy "qpsolvers[proxqp]" \
    rich tqdm opencv-python natsort psutil protobuf "redis[hiredis]" \
    "imageio[ffmpeg]"
"$PIP" install torch "${TORCH_ARGS[@]}"

# ── verify ───────────────────────────────────────────────────────────────────
echo "[setup_gmr] verifying ..."
"$VENV/bin/python" - <<'PYEOF'
import torch
from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
gpu = f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
print(f"  torch {torch.__version__}  CUDA={torch.cuda.is_available()} {gpu}")
print("  GMR BVH/LAFAN import: OK")
PYEOF

echo "[setup_gmr] done."
echo "[setup_gmr] run a video with:  scripts/video_gmr.sh <video.mp4> [RobotType]"
