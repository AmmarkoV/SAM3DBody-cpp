#!/usr/bin/env bash
# tools/setup_trt.sh — one-shot setup for the TensorRT (`--trt`) fast path.
#
# WHAT IT DOES
# ------------
#   1. Creates the tools/.venv and installs TensorRT 10.4 runtime libs
#      (tensorrt-cu12-libs==10.4.0 — the version ORT 1.20.1 was built against).
#      These are what tools/trt_env.sh puts on LD_LIBRARY_PATH so the ORT
#      TensorRT EP can dlopen libnvinfer.so.10 instead of falling back to CUDA.
#   2. Downloads the prebuilt TRT-ready models (backbone_fp16_trt + decoder_fp16,
#      each with its .data sidecar) and unzips them into onnx/.  These are the
#      TRT-buildable variants resolve_backbone_defaults() swaps in under --trt.
#      (The run binaries also fetch these on the fly, but doing it here keeps the
#      first --trt run from blocking on a ~1.7 GB download.)
#
# Idempotent: re-running skips the venv/libs install and the model download when
# they're already present.  Pass --force to redo the download.
#
# Before fetching the ~1.7 GB model archive it asks for confirmation.  In a
# non-interactive shell (no TTY) it refuses unless --yes is given, so it never
# silently pulls a huge file in a script/CI run.  The run binaries invoke this
# (with --skip-venv) on the fly when --trt needs models that aren't on disk.
#
# USAGE
#   tools/setup_trt.sh                  # venv + libs + models (prompts before DL)
#   tools/setup_trt.sh --skip-venv      # only fetch the models
#   tools/setup_trt.sh --yes            # don't prompt (assume yes)
#   tools/setup_trt.sh --force          # re-download the models even if present
#   tools/setup_trt.sh --onnx-dir DIR   # target a non-default models directory
#
# After this, run with TensorRT via:
#   tools/run_trt.sh --onnx-dir ./onnx --from your_video.mp4
# or any of scripts/{webcam,webcam_p,video,offline_video}.sh with --trt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/tools/.venv"
ONNX_DIR="$ROOT/onnx"
TRT_LIBS_PKG="tensorrt-cu12-libs==10.4.0"
MODELS_URL="https://huggingface.co/AmmarkoV/SAM3DBody-cpp-onnx-models/resolve/main/SAM3DBody-cpp-trt-models.zip"

SKIP_VENV=0
FORCE=0
ASSUME_YES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-venv) SKIP_VENV=1; shift ;;
        --force)     FORCE=1;     shift ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        --onnx-dir)  ONNX_DIR="$2"; shift 2 ;;
        --onnx-dir=*) ONNX_DIR="${1#*=}"; shift ;;
        -h|--help)   grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
ZIP="$ONNX_DIR/SAM3DBody-cpp-trt-models.zip"

# ── 1. venv + TensorRT runtime libs ──────────────────────────────────────────
if [ "$SKIP_VENV" -eq 0 ]; then
    echo "=== TensorRT runtime libs (tools/.venv) ==="
    if [ ! -x "$VENV/bin/pip" ]; then
        echo "  Creating venv at $VENV"
        python3 -m venv --upgrade-deps "$VENV"
    fi
    # Already installed?  tensorrt_libs ships the libnvinfer*.so.10 payload.
    if ls "$VENV"/lib/python*/site-packages/tensorrt_libs/libnvinfer.so.10 >/dev/null 2>&1; then
        echo "  $TRT_LIBS_PKG already installed — skipping."
    else
        echo "  Installing $TRT_LIBS_PKG"
        "$VENV/bin/pip" install "$TRT_LIBS_PKG" --extra-index-url https://pypi.nvidia.com
    fi
fi

# ── 2. Prebuilt TRT models → onnx/ ───────────────────────────────────────────
echo "=== Prebuilt TRT models (onnx/) ==="
mkdir -p "$ONNX_DIR"
have_models() {
    [ -f "$ONNX_DIR/backbone_fp16_trt.onnx" ] && \
    [ -f "$ONNX_DIR/backbone_fp16_trt.onnx.data" ] && \
    [ -f "$ONNX_DIR/decoder_fp16.onnx" ] && \
    [ -f "$ONNX_DIR/decoder_fp16.onnx.data" ]
}
if have_models && [ "$FORCE" -eq 0 ]; then
    echo "  backbone_fp16_trt + decoder_fp16 already present — skipping download (use --force to redo)."
else
    # ── Confirm before pulling a large file ──────────────────────────────────
    if [ "$ASSUME_YES" -eq 0 ]; then
        if [ -t 0 ]; then
            printf "  Download prebuilt TRT models (~1.7 GB) into %s ? [y/N] " "$ONNX_DIR"
            read -r ans
            case "$ans" in
                [Yy]|[Yy][Ee][Ss]) ;;
                *) echo "  Skipped — re-run with --yes (or fetch manually per DEPENDENCIES.md §6)."; exit 0 ;;
            esac
        else
            echo "  Refusing to download ~1.7 GB in a non-interactive shell; re-run with --yes." >&2
            exit 1
        fi
    fi

    echo "  Downloading TRT models (~1.7 GB)…"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$ZIP" "$MODELS_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$ZIP" "$MODELS_URL"
    else
        echo "  Need wget or curl on PATH to download the models." >&2
        exit 1
    fi
    # -j flattens the archive's onnx/ prefix so files land directly in onnx/
    # (each .onnx must sit beside its .data sidecar).
    unzip -o -j "$ZIP" -d "$ONNX_DIR"
    rm -f "$ZIP"
    if ! have_models; then
        echo "  Models still missing after unzip — check the archive contents." >&2
        exit 1
    fi
fi

echo
echo "=== TensorRT setup complete ==="
echo "Run with:  tools/run_trt.sh --onnx-dir ./onnx --from your_video.mp4"
echo "       or:  scripts/video.sh --from your_video.mp4 --trt"
