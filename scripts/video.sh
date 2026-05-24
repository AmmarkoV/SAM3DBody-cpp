#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# ── Parse --save [OUTPUT] from the argument list ─────────────────────────────
# --save may be:
#   --save              → save mode; output name derived from source
#   --save output.mp4   → save mode with explicit output name
SAVE_OUTPUT=""
SAVE_REQUESTED=0
FORWARD_ARGS=()
i=1
while [ $i -le $# ]; do
    if [ "${!i}" = "--save" ]; then
        SAVE_REQUESTED=1
        next=$((i+1))
        # Consume next token as output name only if it exists and isn't a flag
        if [ $next -le $# ] && [[ "${!next}" != --* ]]; then
            SAVE_OUTPUT="${!next}"
            i=$next
        fi
    else
        FORWARD_ARGS+=("${!i}")
    fi
    i=$((i+1))
done

# Detect whether --from is already present among the forwarded args.
# If it is, we pass FORWARD_ARGS as-is to the binary; if not, we prepend
# --from (preserving the original single-positional-arg behaviour).
HAS_FROM=0
for _a in "${FORWARD_ARGS[@]}"; do
    [ "$_a" = "--from" ] && HAS_FROM=1 && break
done

FIXED=(
    ./build/fast_sam_3dbody_render
    --onnx-dir ./onnx
    --gguf     ./onnx/pipeline.gguf
    --yolo     ./onnx/yolo.onnx
    --mesh     ./body_mesh.tri
    --lbs      onnx/body_model.lbs
)

if [ "$SAVE_REQUESTED" -eq 0 ]; then
    # ── Normal live/preview mode (original behaviour) ─────────────────────
    if [ "$HAS_FROM" -eq 1 ]; then
        "${FIXED[@]}" "${FORWARD_ARGS[@]}"
    else
        "${FIXED[@]}" --from "${FORWARD_ARGS[@]}"
    fi
    exit $?
fi

# ── Save-to-file mode ─────────────────────────────────────────────────────────

# Locate the source path for ffprobe (FPS + audio detection).
# Prefer an explicit --from VALUE; fall back to the first positional arg.
FROM_SRC=""
_fa=("${FORWARD_ARGS[@]}")
for _i in "${!_fa[@]}"; do
    if [ "${_fa[$_i]}" = "--from" ]; then
        FROM_SRC="${_fa[$((_i+1))]}"
        break
    fi
done
if [ -z "$FROM_SRC" ]; then
    FROM_SRC="${FORWARD_ARGS[0]}"
fi

# Auto-derive output name when --save was given without a value.
# summerlove.mp4 → summerlove_rendered.mp4; fallback: livelastRun3DHiRes.mp4
if [ -z "$SAVE_OUTPUT" ]; then
    if [ -n "$FROM_SRC" ]; then
        base=$(basename "$FROM_SRC")
        stem="${base%.*}"
        SAVE_OUTPUT="${stem}_rendered.mp4"
    else
        SAVE_OUTPUT="livelastRun3DHiRes.mp4"
    fi
    echo "Output: $SAVE_OUTPUT"
fi

# Create a temporary directory for the JPEG frames.
TMPFRAMES=$(mktemp -d /tmp/fsb_frames_XXXXXX)
FRAME_PREFIX="${TMPFRAMES}/colorFrame_0_"

# Run the renderer; it will save every frame as colorFrame_0_NNNNN.jpg.
if [ "$HAS_FROM" -eq 1 ]; then
    "${FIXED[@]}" "${FORWARD_ARGS[@]}" --save-frames "$FRAME_PREFIX"
else
    "${FIXED[@]}" --from "${FORWARD_ARGS[@]}" --save-frames "$FRAME_PREFIX"
fi
RENDER_EXIT=$?
if [ $RENDER_EXIT -ne 0 ]; then
    echo "Renderer exited with code $RENDER_EXIT — aborting encode." >&2
    rm -rf "$TMPFRAMES"
    exit $RENDER_EXIT
fi

# ── Probe the source for frame rate ──────────────────────────────────────────
FPS=30
if [ -n "$FROM_SRC" ] && [ -f "$FROM_SRC" ]; then
    RAW_FPS=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "$FROM_SRC" 2>/dev/null)
    if [ -n "$RAW_FPS" ]; then
        FPS=$(awk -F'/' 'NF==2 && $2>0 { printf "%g\n", $1/$2; next } { print $1 }' \
              <<< "$RAW_FPS")
    fi
fi
[ -z "$FPS" ] && FPS=30
echo "Source framerate: ${FPS} fps"

# ── Get rendered frame size from the first JPEG ───────────────────────────────
SIZE_ARG=()
FIRST_FRAME=$(ls "${FRAME_PREFIX}"*.jpg 2>/dev/null | sort | head -1)
if [ -n "$FIRST_FRAME" ]; then
    read -r FW FH < <(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=p=0 "$FIRST_FRAME" 2>/dev/null | \
        tr ',' ' ')
    if [[ "$FW" =~ ^[0-9]+$ ]] && [[ "$FH" =~ ^[0-9]+$ ]]; then
        # yuv420p requires even dimensions
        FW=$(( (FW / 2) * 2 ))
        FH=$(( (FH / 2) * 2 ))
        SIZE_ARG=(-s "${FW}x${FH}")
        echo "Render size: ${FW}x${FH}"
    fi
fi

# ── Check for audio in the source ────────────────────────────────────────────
AUDIO_ARGS=()
if [ -n "$FROM_SRC" ] && [ -f "$FROM_SRC" ]; then
    AUDIO_IDX=$(ffprobe -v error -select_streams a:0 \
        -show_entries stream=index -of csv=p=0 "$FROM_SRC" 2>/dev/null)
    if [ -n "$AUDIO_IDX" ]; then
        echo "Copying audio from: $FROM_SRC"
        AUDIO_ARGS=(-i "$FROM_SRC" -map 0:v -map 1:a -c:a copy)
    fi
fi

# ── Encode ────────────────────────────────────────────────────────────────────
ffmpeg -framerate "$FPS" \
       -i "${FRAME_PREFIX}%05d.jpg" \
       "${AUDIO_ARGS[@]}" \
       "${SIZE_ARG[@]}" \
       -y -r "$FPS" -pix_fmt yuv420p -threads 8 \
       "$SAVE_OUTPUT"
FFMPEG_EXIT=$?

# ── Clean up JPEG frames ──────────────────────────────────────────────────────
rm -f "${TMPFRAMES}"/colorFrame_0_*.jpg
rmdir "$TMPFRAMES"

exit $FFMPEG_EXIT
