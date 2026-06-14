#!/usr/bin/env bash
# fast_sam_3dbody_cpp/setup.sh
#
# 1. Download ONNX/GGUF/LBS models from HuggingFace (into onnx/).
# 2. Build the C++ shared library and CLI (cmake + make).
# 3. Create a Python venv with the packages required by both Python frontends:
#      fast_sam_3dbody_frontend.py      – needs only opencv-python + numpy
#      fast_sam_3dbody_frontend-3D.py   – also needs torch, pyrender, roma, etc.
#
# Usage (from repo root or from fast_sam_3dbody_cpp/):
#   bash scripts/setup.sh
#   bash scripts/setup.sh --cuda-arch 89        # RTX 4090
#   bash scripts/setup.sh --ort-dir /opt/onnxruntime
#   bash scripts/setup.sh --cpu-only            # no CUDA torch
#   bash scripts/setup.sh --cpu-backbone        # download fp32 backbone (CPU inference)
#   bash scripts/setup.sh --skip-models         # skip model download
#   bash scripts/setup.sh --skip-build          # venv only
#   bash scripts/setup.sh --skip-venv           # build only
#   bash scripts/setup.sh -j 4                  # limit make jobs

set -euo pipefail

# ── System dependency check ────────────────────────────────────────────────────
# Simple dependency checker that will apt-get stuff if something is missing.
# Note: wget and unzip are now OPTIONAL — the new tools.py wrappers use the
# Python stdlib (urllib + zipfile) first and only fall back to these binaries
# if the stdlib path fails. They remain in the list so existing scripts under
# scripts/ that shell out directly (downloadPretrained.sh etc.) keep working.
SYSTEM_DEPENDENCIES="python3-pip python3-venv build-essential cmake git wget unzip \
    libjpeg-dev libpng-dev libzstd-dev liblz4-dev libpthread-stubs0-dev \
    libopencv-dev libgl1-mesa-dev"

for REQUIRED_PKG in $SYSTEM_DEPENDENCIES; do
    PKG_OK=$(dpkg-query -W --showformat='${Status}\n' "$REQUIRED_PKG" 2>/dev/null | grep "install ok installed" || true)
    echo "Checking for $REQUIRED_PKG: ${PKG_OK:-missing}"
    if [ "" = "$PKG_OK" ]; then
        echo "No $REQUIRED_PKG. Installing all missing dependencies now …"
        sudo apt-get install -y $SYSTEM_DEPENDENCIES
        break
    fi
done

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"
BUILD_DIR="${REPO_ROOT}/build"
VENV_DIR="${REPO_ROOT}/venv"
ONNX_DIR="${REPO_ROOT}/onnx"

HF_BASE="https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main"
HF_ZIP_NAME="SAM3DBody-cpp-onnx-models.zip"
HF_ZIP_URL="${HF_BASE}/${HF_ZIP_NAME}"

# ── Defaults ───────────────────────────────────────────────────────────────────
CUDA_ARCH=""
ORT_DIR=""
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
JOBS=$(( ($(nproc 2>/dev/null || echo 4) + 1) / 2 ))   # half the cores to keep system responsive
SKIP_BUILD=0
SKIP_VENV=0
SKIP_MODELS=0
CPU_BACKBONE=0

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-arch)    CUDA_ARCH="$2";            shift 2 ;;
        --cuda-arch=*)  CUDA_ARCH="${1#*=}";       shift ;;
        --ort-dir)      ORT_DIR="$2";              shift 2 ;;
        --ort-dir=*)    ORT_DIR="${1#*=}";         shift ;;
        --cpu-only)     TORCH_INDEX="https://download.pytorch.org/whl/cpu"; shift ;;
        --cpu-backbone) CPU_BACKBONE=1;            shift ;;
        --skip-models)  SKIP_MODELS=1;             shift ;;
        --skip-build)   SKIP_BUILD=1;              shift ;;
        --skip-venv)    SKIP_VENV=1;               shift ;;
        -j)             JOBS="$2";                 shift 2 ;;
        -j*)            JOBS="${1#-j}";            shift ;;
        -h|--help)
            sed -n '2,/^[^#]/{ /^[^#]/q; s/^# \?//p }' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Helper: download with resume support ───────────────────────────────────────
_download() {
    local url="$1" dest="$2" desc="${3:-}"
    [[ -n "$desc" ]] && echo "  Downloading ${desc} …"
    if command -v wget &>/dev/null; then
        wget --continue --show-progress -q -O "${dest}" "${url}"
    elif command -v curl &>/dev/null; then
        curl -L --continue-at - --progress-bar -o "${dest}" "${url}"
    else
        echo "ERROR: neither wget nor curl found. Install one and retry." >&2
        exit 1
    fi
}

# ── Check for required model files ────────────────────────────────────────────
_models_present() {
    local required=( backbone.onnx decoder.onnx yolo.onnx pipeline.gguf body_model.lbs )
    for f in "${required[@]}"; do
        [[ -f "${ONNX_DIR}/${f}" ]] || return 1
    done
    return 0
}

# ── Model download ─────────────────────────────────────────────────────────────
if [[ "${SKIP_MODELS}" -eq 0 ]]; then
    echo "=== Checking models in ${ONNX_DIR} ==="
    if _models_present; then
        echo "  All required model files found — skipping download."
        echo "  (Pass --skip-models to suppress this check, or delete onnx/ to re-download.)"
    else
        echo "  Models not found. Downloading from HuggingFace (~5 GB)…"
        echo "  Source: ${HF_ZIP_URL}"
        echo ""

        mkdir -p "${ONNX_DIR}"
        _ZIP_TMP="${REPO_ROOT}/${HF_ZIP_NAME}"

        if [[ -f "${_ZIP_TMP}" ]]; then
            echo "  Found existing partial/complete zip at ${_ZIP_TMP} — attempting resume."
        fi

        _download "${HF_ZIP_URL}" "${_ZIP_TMP}" "${HF_ZIP_NAME}"

        echo "  Extracting ${HF_ZIP_NAME} …"
        if command -v unzip &>/dev/null; then
            unzip -o "${_ZIP_TMP}" -d "${REPO_ROOT}"
        else
            echo "ERROR: unzip not found. Install it (e.g. sudo apt install unzip) and retry." >&2
            exit 1
        fi

        # The zip extracts to onnx/ at repo root
        if _models_present; then
            echo "  Extraction OK. Removing zip archive …"
            rm -f "${_ZIP_TMP}"
        else
            echo "WARNING: Extraction finished but some model files are still missing." >&2
            echo "         Zip kept at ${_ZIP_TMP} for inspection." >&2
        fi
    fi

    # Optional: CPU-compatible fp32 backbone (for machines without CUDA)
    if [[ "${CPU_BACKBONE}" -eq 1 ]]; then
        echo ""
        echo "=== Downloading fp32 backbone (CPU inference) ==="
        for _f in backbone_fp32.onnx "backbone_fp32.onnx.data"; do
            if [[ -f "${ONNX_DIR}/${_f}" ]]; then
                echo "  ${_f} already present — skipping."
            else
                _download "${HF_BASE}/${_f}" "${ONNX_DIR}/${_f}" "${_f}"
            fi
        done
        echo ""
        echo "  To use the CPU backbone, run:"
        echo "    ./build/fast_sam_3dbody_run --onnx-dir ./onnx --backbone backbone_fp32.onnx --cuda -1 --from <input>"
    fi
fi

# ── External dependency: RGBDAcquisition ───────────────────────────────────────
_RGBDA_REPO="https://github.com/AmmarkoV/RGBDAcquisition"
_RGBDA_DIR="${REPO_ROOT}/RGBDAcquisition"

echo ""
echo "=== Setting up RGBDAcquisition ==="
if [[ -d "${_RGBDA_DIR}/.git" ]]; then
    echo "  Repository already exists — pulling latest …"
    git -C "${_RGBDA_DIR}" pull --ff-only \
        || echo "  Warning: git pull failed; continuing with existing checkout."
else
    echo "  Cloning ${_RGBDA_REPO} …"
    git clone "${_RGBDA_REPO}" "${_RGBDA_DIR}"
fi

_make_symlink() {
    local link="$1" target="$2"
    if [[ -L "${link}" ]]; then
        echo "  Symlink '${link}' already exists — skipping."
    elif [[ -e "${link}" ]]; then
        echo "  Warning: '${link}' exists but is not a symlink — skipping."
    else
        ln -s "${target}" "${link}"
        echo "  Created: ${link} → ${target}"
    fi
    if [[ ! -e "${link}" ]]; then
        echo "  Warning: '${link}' target does not exist yet (run after a full clone)."
    fi
}

_make_symlink "${REPO_ROOT}/GraphicsEngine" \
    "${_RGBDA_DIR}/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library"
_make_symlink "${REPO_ROOT}/AmMatrix" \
    "${_RGBDA_DIR}/tools/AmMatrix"


# ── C++ build ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo ""
    echo "=== Building C++ library and CLI ==="
    mkdir -p "${BUILD_DIR}"

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    [[ -n "${CUDA_ARCH}" ]] && CMAKE_ARGS+=" -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    [[ -n "${ORT_DIR}" ]]   && CMAKE_ARGS+=" -DONNX_RUNTIME_DIR=${ORT_DIR}"

    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" ${CMAKE_ARGS}
    cmake --build "${BUILD_DIR}" -- -j"${JOBS}"

    echo ""
    echo "Build outputs:"
    for f in "${BUILD_DIR}/libfast_sam_3dbody.so" "${BUILD_DIR}/fast_sam_3dbody_run"; do
        [[ -f "$f" ]] && echo "  $(du -sh "$f" | cut -f1)  $f" || echo "  [missing] $f"
    done
fi

# ── Python venv ────────────────────────────────────────────────────────────────
if [[ "${SKIP_VENV}" -eq 0 ]]; then
    echo ""
    echo "=== Creating Python venv at ${VENV_DIR} ==="

    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip --quiet

    echo "--- Core (both frontends) ---"
    pip install numpy opencv-python

    echo "--- 3D frontend: torch ---"
    pip install torch torchvision --index-url "${TORCH_INDEX}"

    echo "--- 3D frontend: sam_3d_body deps ---"
    # pyrender + trimesh: 3D mesh rendering
    # roma: rotation representations
    # einops: tensor ops used in backbone/decoder modules
    # timm: ViT layer utilities (drop_path, trunc_normal_)
    # omegaconf + yacs: config loading in sam_3d_body.utils.config
    # huggingface_hub: checkpoint auto-download on first run
    pip install pyrender trimesh roma einops timm omegaconf yacs huggingface_hub braceexpand pytorch_lightning termcolor

    deactivate

    echo ""
    echo "Venv ready: ${VENV_DIR}"
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo ""
echo "  # Live webcam (CUDA):"
echo "  ./scripts/webcam.sh"
echo ""
echo "  # Process a video file:"
echo "  ./scripts/video.sh --from your_video.mp4"
echo ""
echo "  # Activate the venv for Python frontends:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "  # Lightweight frontend (2D skeleton only):"
echo "  python fast_sam_3dbody_frontend.py --from assets/teaser.png"
echo ""
echo "  # 3D frontend (full mesh rendering):"
echo "  python fast_sam_3dbody_frontend-3D.py --from assets/teaser.png"
