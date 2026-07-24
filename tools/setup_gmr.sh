#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  setup_gmr.sh
#
#  Sets up the GMR (General Motion Retargeting) Python environment used by
#  scripts/video_gmr.sh to turn the LAFAN BVH we export into humanoid-robot
#  motion.
#
#  GMR itself lives at  $REPO/GMR.  If it is not there, this script clones it
#  (https://github.com/YanjieZe/GMR; override with GMR_REPO=...); an existing
#  checkout or symlink is reused untouched.  It then creates a venv inside it and
#  installs the runtime deps:
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

# ── GMR: clone it if not already present (symlink or checkout) ───────────────
GMR_REPO="${GMR_REPO:-https://github.com/YanjieZe/GMR}"
if [ ! -f "$GMR_DIR/setup.py" ]; then
    if [ -e "$GMR_DIR" ]; then
        echo "ERROR: $GMR_DIR exists but has no setup.py (not a GMR checkout)." >&2
        exit 1
    fi
    echo "[setup_gmr] GMR not found at $GMR_DIR — cloning $GMR_REPO"
    git clone --depth 1 "$GMR_REPO" "$GMR_DIR"
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

# ── SharedMemoryVideoBuffers: the live webcam→robot zero-copy transport ──────
# scripts/webcam_gmr.sh can hand each frame's BVH channels to gmr_stream.py over
# POSIX shared memory instead of an ASCII stdout pipe + per-frame temp .bvh file.
# The C++ binary links this library's C source at build time (CMake picks it up
# automatically when this checkout exists — Linux only); gmr_stream.py dlopens
# the .so we build here via ctypes.  Clone + build are best-effort: if either
# fails the pipeline silently falls back to the stdout/tempfile path.
SHM_DIR="$REPO/SharedMemoryVideoBuffers"
SHM_REPO="${SHM_REPO:-https://github.com/AmmarkoV/SharedMemoryVideoBuffers}"
if [ "$(uname -s)" = "Linux" ]; then
    if [ ! -f "$SHM_DIR/src/c/sharedMemoryVideoBuffers.c" ]; then
        if [ -e "$SHM_DIR" ]; then
            echo "[setup_gmr] WARN: $SHM_DIR exists but isn't a SharedMemoryVideoBuffers checkout — skipping shm setup" >&2
        else
            echo "[setup_gmr] cloning SharedMemoryVideoBuffers (live shm transport) — $SHM_REPO"
            git clone --depth 1 "$SHM_REPO" "$SHM_DIR" || \
                echo "[setup_gmr] WARN: clone failed — webcam_gmr.sh will use the stdout/tempfile fallback" >&2
        fi
    fi
    if [ -f "$SHM_DIR/src/c/sharedMemoryVideoBuffers.c" ]; then
        echo "[setup_gmr] building libSharedMemoryVideoBuffers.so"
        if make -C "$SHM_DIR" libSharedMemoryVideoBuffers.so >/dev/null 2>&1 \
           || make -C "$SHM_DIR" >/dev/null 2>&1; then
            echo "[setup_gmr] shm library ready: $SHM_DIR/libSharedMemoryVideoBuffers.so"
            echo "[setup_gmr] NOTE: (re)build the C++ binary after this so it picks up shm support (scripts/build.sh)"
        else
            echo "[setup_gmr] WARN: shm library build failed — webcam_gmr.sh will use the stdout/tempfile fallback" >&2
        fi
    fi
else
    echo "[setup_gmr] non-Linux host — skipping SharedMemoryVideoBuffers (shm transport is Linux-only)"
fi

# ── generate the LAFAN template with the MHR rest pose ───────────────────────
# video_gmr.sh retargets onto lafan_mhr.bvh — a LAFAN1-named template whose REST
# pose equals the MHR rest skeleton (so the BVH writer's q_bone_align is identity
# and joint world positions are faithful).  Needs onnx/body_model.lbs.  See
# tools/gen_lafan_bvh.py and GMR.md.
if [ -f "$REPO/onnx/body_model.lbs" ]; then
    echo "[setup_gmr] generating lafan_mhr.bvh (MHR rest pose)"
    # non-fatal: a generation hiccup must not abort the whole env setup
    "$PY" "$THISDIR/gen_lafan_bvh.py" || \
        echo "[setup_gmr] WARN: gen_lafan_bvh.py failed — run it manually before video_gmr.sh" >&2
else
    echo "[setup_gmr] WARN: onnx/body_model.lbs missing — skipping lafan_mhr.bvh;" >&2
    echo "           run 'python3 tools/gen_lafan_bvh.py' once the model is present." >&2
fi

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
