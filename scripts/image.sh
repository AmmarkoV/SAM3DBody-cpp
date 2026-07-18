#!/bin/bash
# scripts/image.sh FILE [FILE ...] [extra flags]
#
# Render the 3D mesh overlay for still images.  For each input the output is
# the stem plus _v, keeping the original extension:
#
#   xoroi.jpeg  →  xoroi_v.jpeg
#   crowd.png   →  crowd_v.png
#
# The renderer's --save-frames always appends a %05d index, so this writes the
# frames to a temp dir and moves frame 00001 into place.
#
# Anything after the first --flag argument is forwarded verbatim to every
# renderer call:
#
#   scripts/image.sh xoroi.jpeg
#   scripts/image.sh *.jpg
#   scripts/image.sh xoroi.jpeg --cuda -1 --thresh 0.6

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

# ── Split arguments into input files and forwarded flags ─────────────────────
FILES=()
FORWARD_ARGS=()
for arg in "$@"; do
    if [ ${#FORWARD_ARGS[@]} -gt 0 ] || [[ "$arg" == --* ]]; then
        FORWARD_ARGS+=("$arg")
    else
        FILES+=("$arg")
    fi
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Usage: scripts/image.sh FILE [FILE ...] [extra flags]" >&2
    exit 1
fi

BIN=./build/fast_sam_3dbody_render
FIXED_FLAGS=(
    --onnx-dir ./onnx
    --gguf     ./onnx/pipeline.gguf
    --yolo     ./onnx/yolo.onnx
    --mesh     ./body_mesh.tri
    --lbs      onnx/body_model.lbs
)

STATUS=0
for SRC in "${FILES[@]}"; do
    if [ ! -f "$SRC" ]; then
        echo "Skipping missing file: $SRC" >&2
        STATUS=1
        continue
    fi

    base=$(basename "$SRC")
    stem="${base%.*}"
    ext="${base##*.}"
    # The renderer only writes JPEG, so keep .jpg/.jpeg as-is and normalise
    # anything else (png, bmp, …) to .jpg rather than mislabelling the file.
    shopt -s nocasematch
    [[ "$ext" == jpg || "$ext" == jpeg ]] || ext="jpg"
    shopt -u nocasematch
    OUT="$(dirname "$SRC")/${stem}_v.${ext}"

    # Match the render surface to the image so the result isn't letterboxed
    # into the renderer's default 16:9 window.  Skipped if ffprobe is absent.
    SIZE_ARG=()
    if command -v ffprobe &>/dev/null; then
        read -r IW IH < <(ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height -of csv=p=0 "$SRC" 2>/dev/null | tr ',' ' ')
        if [[ "$IW" =~ ^[0-9]+$ ]] && [[ "$IH" =~ ^[0-9]+$ ]]; then
            SIZE_ARG=(--render-size "$IW" "$IH")
        fi
    fi

    TMPFRAMES=$(mktemp -d /tmp/fsb_image_XXXXXX)
    FRAME_PREFIX="${TMPFRAMES}/frame_"

    # --headless renders into an offscreen GLX Pbuffer, so no window pops up
    # for what is a single-frame run.
    "$BIN" --from "$SRC" "${FIXED_FLAGS[@]}" "${SIZE_ARG[@]}" \
           "${FORWARD_ARGS[@]}" --headless --save-frames "$FRAME_PREFIX"

    FRAME=$(ls "${FRAME_PREFIX}"*.jpg 2>/dev/null | sort | head -1)
    if [ -n "$FRAME" ]; then
        mv "$FRAME" "$OUT"
        echo "Wrote: $OUT"
    else
        echo "No frame rendered for $SRC" >&2
        STATUS=1
    fi

    rm -rf "$TMPFRAMES"
done

exit $STATUS
