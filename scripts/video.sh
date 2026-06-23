#!/bin/bash

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Activate project venv and put TensorRT libs on LD_LIBRARY_PATH (both no-op if not set up).
source "$THISDIR/../tools/project_env.sh"
source "$THISDIR/../tools/trt_env.sh"

# ── Parse --save [OUTPUT] from the argument list ─────────────────────────────
# --save may be:
#   --save              → save mode; output name derived from source
#   --save output.mp4   → save mode with explicit output name
SAVE_OUTPUT=""
SAVE_REQUESTED=0
FROM_SRC=""
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
    elif [ "${!i}" = "--from" ]; then
        # Resolve the input here and strip it; it is always re-emitted FIRST
        # (right after the binary) so the filename is visible at the front of
        # the command line in htop/ps.
        next=$((i+1))
        if [ $next -le $# ]; then FROM_SRC="${!next}"; i=$next; fi
    else
        FORWARD_ARGS+=("${!i}")
    fi
    i=$((i+1))
done

# Positional fallback: if no --from was given, the first non-flag forwarded
# token is the source (preserves the original single-positional-arg style,
# e.g. `video.sh /dev/video0`).
if [ -z "$FROM_SRC" ] && [ ${#FORWARD_ARGS[@]} -gt 0 ] && [[ "${FORWARD_ARGS[0]}" != --* ]]; then
    FROM_SRC="${FORWARD_ARGS[0]}"
    FORWARD_ARGS=("${FORWARD_ARGS[@]:1}")
fi

# --from "$FROM_SRC" is emitted FIRST, right after the binary, so the input
# filename leads the command line in htop/ps.
BIN=./build/fast_sam_3dbody_render
FIXED_FLAGS=(
    --onnx-dir ./onnx
    --gguf     ./onnx/pipeline.gguf
    --yolo     ./onnx/yolo.onnx
    --mesh     ./body_mesh.tri
    --lbs      onnx/body_model.lbs
    # --enforce-hand-limits and --sticky-hand-pose are now ON by default;
    # pass --no-enforce-hand-limits / --no-sticky-hand-pose to disable.
)

if [ "$SAVE_REQUESTED" -eq 0 ]; then
    # ── Normal live/preview mode (original behaviour) ─────────────────────
    "$BIN" --from "$FROM_SRC" "${FIXED_FLAGS[@]}" "${FORWARD_ARGS[@]}"
    exit $?
fi

# ── Save-to-file mode ─────────────────────────────────────────────────────────

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
#
# --headless is forced in --save mode so the long-running render uses an
# offscreen GLX Pbuffer instead of a visible X11 window.  Without it, a
# user (or screen-saver, or session lock) closing the window mid-render
# would terminate the renderer with whatever JPEGs had been written so
# far — and the encode step below would happily produce a frozen-frame
# mp4 from the partial output.  This was the cause of the
# "matrix_rendered.mp4 freezes at 0:17 with correct audio length"
# regression: the renderer was killed before reaching the end and the
# encode silently used the truncated frame set.
HEADLESS_ARG=("--headless")
"$BIN" --from "$FROM_SRC" "${FIXED_FLAGS[@]}" "${FORWARD_ARGS[@]}" "${HEADLESS_ARG[@]}" --save-frames "$FRAME_PREFIX"
RENDER_EXIT=$?

# ── Validate the rendered frame count ─────────────────────────────────────────
# The exit code alone is unreliable: the renderer is known to raise harmless
# late-stage cleanup errors (e.g. Pbuffer-mode XDestroyWindow → BadWindow)
# that surface as a non-zero exit AFTER every frame has been written.  We'd
# rather encode-and-warn in that case than throw away a complete frame set.
#
# So the abort rule is: only refuse to encode when the JPEGs themselves are
# short of what the source video has.  When reading from a video file the
# expected count is ffprobe nb_frames; webcams have no fixed length so we
# skip the check entirely and trust the renderer's exit code there.
ACTUAL=$(ls "${FRAME_PREFIX}"*.jpg 2>/dev/null | wc -l)
ENCODE_OK=1
if [ -n "$FROM_SRC" ] && [ -f "$FROM_SRC" ]; then
    EXPECTED=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=nb_frames -of csv=p=0 "$FROM_SRC" 2>/dev/null)
    if [[ "$EXPECTED" =~ ^[0-9]+$ ]] && [ "$EXPECTED" -gt 0 ]; then
        # Tolerate small slack — some containers report off-by-one nb_frames,
        # and the renderer's probe consumes 1 frame before the loop starts.
        MIN_OK=$(( EXPECTED - 5 ))
        if [ "$ACTUAL" -lt "$MIN_OK" ]; then
            echo "Renderer wrote $ACTUAL frames, source has $EXPECTED — aborting encode." >&2
            echo "(The renderer was killed mid-run.  Re-run; if it persists, try --skip-body.)" >&2
            ENCODE_OK=0
        fi
    fi
elif [ $RENDER_EXIT -ne 0 ]; then
    # Webcam / live source — defer to the renderer's exit code.
    ENCODE_OK=0
fi

if [ "$ENCODE_OK" -ne 1 ]; then
    rm -rf "$TMPFRAMES"
    exit ${RENDER_EXIT:-2}
fi

if [ $RENDER_EXIT -ne 0 ]; then
    echo "Renderer exited with code $RENDER_EXIT but all $ACTUAL frames are present — encoding anyway." >&2
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
