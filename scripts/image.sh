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
#
# --exif-focal (needs exiftool) reads the real camera focal length out of the
# image and passes it as --fx/--fy instead of letting the pipeline use its
# default.  OFF by default on purpose: that default is the image diagonal,
# which is the focal the decoder/FFN was trained against (see the comment at
# src/fast_sam_3dbody.cpp:610), so a true camera focal changes condition_info
# and can just as easily make the fit worse.  Treat it as an experiment.

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

# ── Split arguments into input files and forwarded flags ─────────────────────
FILES=()
FORWARD_ARGS=()
EXIF_FOCAL=0
for arg in "$@"; do
    if [ "$arg" = "--exif-focal" ]; then
        EXIF_FOCAL=1
    elif [ ${#FORWARD_ARGS[@]} -gt 0 ] || [[ "$arg" == --* ]]; then
        FORWARD_ARGS+=("$arg")
    else
        FILES+=("$arg")
    fi
done

if [ "$EXIF_FOCAL" -eq 1 ] && ! command -v exiftool &>/dev/null; then
    echo "--exif-focal needs exiftool (sudo apt install libimage-exiftool-perl)" >&2
    exit 1
fi

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

    # ── Optional EXIF-derived intrinsics ─────────────────────────────────────
    # fx_pixels = focal_35mm_equivalent / 36 mm * image_width.  The 35mm-equiv
    # tag is used rather than the raw focal length because it already folds in
    # the sensor crop factor, so no per-body sensor-size table is needed.
    FOCAL_ARG=()
    if [ "$EXIF_FOCAL" -eq 1 ]; then
        read -r F35 EW MODEL < <(exiftool -s3 -n \
            -FocalLengthIn35mmFormat -ImageWidth -Model "$SRC" 2>/dev/null | tr '\n' ' ')
        if [[ "$F35" =~ ^[0-9.]+$ ]] && [[ "$EW" =~ ^[0-9]+$ ]] && [ "$EW" -gt 0 ]; then
            FX=$(awk -v f="$F35" -v w="$EW" 'BEGIN { printf "%.1f", f / 36.0 * w }')
            FOCAL_ARG=(--fx "$FX" --fy "$FX")
            echo "EXIF: ${MODEL:-unknown camera}, ${F35}mm eq → fx=${FX}px"
        else
            echo "No usable EXIF focal in $SRC — using pipeline default" >&2
        fi
    fi

    TMPFRAMES=$(mktemp -d /tmp/fsb_image_XXXXXX)
    FRAME_PREFIX="${TMPFRAMES}/frame_"

    # --headless renders into an offscreen GLX Pbuffer, so no window pops up
    # for what is a single-frame run.
    "$BIN" --from "$SRC" "${FIXED_FLAGS[@]}" "${SIZE_ARG[@]}" "${FOCAL_ARG[@]}" \
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
