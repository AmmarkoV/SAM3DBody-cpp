#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  offline_video.sh
#
#  Entry point for the offline multi-pass BVH extractor.
#
#  Inherits the same conventions as scripts/video.sh:
#    - "$THISDIR/.." is the project root
#    - --bvh / --bvh-template / --no-bvh-*-shape-change / --bvh-raw-fingers
#      flags are forwarded unchanged
#    - --from VIDEO is the input
#
#  Differences vs scripts/video.sh:
#    - calls ./build/offline_sam_3dbody_render (no OpenGL preview)
#    - --from MUST be a video file (the binary refuses webcam indices and
#      single images)
#    - additional offline-only flags pass through:
#         --smoothing zero-phase|forward|off
#         --interpolate-jitter --jitter-threshold-cm
#         --track-merge-frames --track-merge-cm
#         --static-scene
#
#  Optional rendering of an annotated mp4 alongside the BVH:
#    - `--save [OUT.mp4]` makes this script run scripts/video.sh AFTER the
#      offline binary finishes, using the same --from input.  Two files
#      are produced: BVH from the offline pipeline (smoother / better
#      identities) AND a visualisation mp4 from the live renderer (per-
#      frame inference, no offline smoothing).  This delegation keeps the
#      rendering path single-sourced in scripts/video.sh.
#
#  Usage:
#     scripts/offline_video.sh --from clip.mp4 --bvh out.bvh [--smoothing ...]
#     scripts/offline_video.sh --from clip.mp4 --bvh out.bvh --save vis.mp4
# ════════════════════════════════════════════════════════════════════════════

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"
cd ..

# Locate the input video — accept the same forms scripts/video.sh does:
# either `--from PATH` or a single positional path argument.
FROM_SRC=""
HAS_FROM=0
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--from" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        FROM_SRC="${ARGS[$((i+1))]}"
        HAS_FROM=1
        break
    fi
done
if [ $HAS_FROM -eq 0 ] && [ -n "${ARGS[0]:-}" ] && [ "${ARGS[0]:0:1}" != "-" ]; then
    FROM_SRC="${ARGS[0]}"
    ARGS=("${ARGS[@]:1}")   # drop the positional so we can prepend --from
fi

if [ -z "$FROM_SRC" ]; then
    echo "Usage: $(basename "$0") --from VIDEO --bvh OUT.bvh [options] [--save VIS.mp4]" >&2
    echo "       (or positional: $(basename "$0") VIDEO --bvh OUT.bvh ...)" >&2
    exit 2
fi
if [ ! -f "$FROM_SRC" ]; then
    echo "input video not found: $FROM_SRC" >&2
    exit 2
fi

# ── Extract --save from ARGS ─────────────────────────────────────────────────
# The offline binary itself does NOT know about --save (it produces only
# BVH).  We catch the flag here and, if set, delegate the rendered-mp4 step
# to scripts/video.sh AFTER the offline run completes.  The offline binary
# then sees a clean argv that contains only flags it actually understands.
SAVE_REQUESTED=0
SAVE_OUTPUT=""
OFFLINE_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    a="${ARGS[$i]}"
    if [ "$a" = "--save" ]; then
        SAVE_REQUESTED=1
        next=$((i+1))
        # Same convention as video.sh: optional value, only consumed when
        # it's a non-flag token.
        if [ $next -lt ${#ARGS[@]} ] && [[ "${ARGS[$next]}" != --* ]]; then
            SAVE_OUTPUT="${ARGS[$next]}"
            i=$next
        fi
    elif [ "$a" = "--from" ]; then
        # --from is already resolved into $FROM_SRC above and is always
        # re-emitted FIRST (see the exec below) so the input filename shows
        # up at the front of the command line in htop/ps.  Strip it (and its
        # value) here to avoid passing it twice.
        next=$((i+1))
        [ $next -lt ${#ARGS[@]} ] && i=$next
    else
        OFFLINE_ARGS+=("$a")
    fi
    i=$((i+1))
done

# ── Run the offline binary ───────────────────────────────────────────────────
# --from "$FROM_SRC" is emitted FIRST, right after the binary, so the input
# filename is visible at the front of the command line in htop/ps — makes it
# obvious which job is running during a long batch.
BIN=./build/offline_sam_3dbody_render
FIXED_FLAGS=(
    --onnx-dir ./onnx
    --gguf     ./onnx/pipeline.gguf
    --yolo     ./onnx/yolo.onnx
    # --enforce-hand-limits and --sticky-hand-pose are now ON by default;
    # pass --no-enforce-hand-limits / --no-sticky-hand-pose to disable.
)
"$BIN" --from "$FROM_SRC" "${FIXED_FLAGS[@]}" "${OFFLINE_ARGS[@]}"
OFFLINE_EXIT=$?
if [ $OFFLINE_EXIT -ne 0 ]; then
    echo "offline binary exited with code $OFFLINE_EXIT — skipping rendered-mp4 step." >&2
    exit $OFFLINE_EXIT
fi

# ── Optional render-mp4 delegation ───────────────────────────────────────────
# Hand off to scripts/video.sh, which already does --save-frames →
# ffmpeg encoding + audio copy.  We only pass --from / --save (and the
# explicit save name if given) — none of the offline-only flags are
# meaningful to the live renderer.
if [ $SAVE_REQUESTED -eq 1 ]; then
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo " offline BVH done — now rendering visualisation mp4 via video.sh"
    echo "──────────────────────────────────────────────────────────────────"
    if [ -n "$SAVE_OUTPUT" ]; then
        "$THISDIR/video.sh" --from "$FROM_SRC" --save "$SAVE_OUTPUT"
    else
        "$THISDIR/video.sh" --from "$FROM_SRC" --save
    fi
fi
