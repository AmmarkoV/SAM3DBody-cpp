#!/usr/bin/env bash
# scripts/batch_video.sh PATTERN [extra flags]
#
# Run scripts/video.sh --save --bvh on every file matching PATTERN, serially.
#
# For each matched file the derived outputs are:
#   borat.mkv  →  --from borat.mkv  --save borat_rendered.mp4  --bvh borat.bvh
#
# Any extra flags are forwarded verbatim to every video.sh invocation.
#
# Usage:
#   scripts/batch_video.sh "*.mkv"
#   scripts/batch_video.sh "clips/*.mp4" --skip-body --butterworth
#   scripts/batch_video.sh "*.mkv" --cuda -1 --bw-cutoff 4.0
#
# The PATTERN must be quoted to prevent the shell from expanding it before
# this script can do it relative to the working directory.

set -euo pipefail

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_SH="$THISDIR/video.sh"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") PATTERN [extra flags for video.sh]"
    echo "  Example: $(basename "$0") \"*.mkv\" --butterworth"
    exit 1
fi

PATTERN="$1"
shift
EXTRA=("$@")

# Expand the glob now (in the caller's working directory).
shopt -s nullglob
FILES=($PATTERN)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files matched: $PATTERN"
    exit 1
fi

echo "Batch: ${#FILES[@]} file(s) matched '$PATTERN'"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "Extra flags: ${EXTRA[*]}"
echo

PASS=0
FAIL=0
FAILED_FILES=()

for FILE in "${FILES[@]}"; do
    base=$(basename "$FILE")
    stem="${base%.*}"

    BVH_OUT="${stem}.bvh"
    RENDERED_OUT="${stem}_rendered.mp4"

    echo "──────────────────────────────────────────────"
    echo "  Input:    $FILE"
    echo "  BVH:      $BVH_OUT"
    echo "  Rendered: $RENDERED_OUT"
    echo "──────────────────────────────────────────────"

    if bash "$VIDEO_SH" \
            --from "$FILE" \
            --save "$RENDERED_OUT" \
            --bvh  "$BVH_OUT" \
            "${EXTRA[@]}"; then
        PASS=$((PASS + 1))
        echo "  ✓ done: $FILE"
    else
        FAIL=$((FAIL + 1))
        FAILED_FILES+=("$FILE")
        echo "  ✗ FAILED: $FILE (continuing)"
    fi
    echo
done

echo "══════════════════════════════════════════════"
echo "  Batch complete: $PASS succeeded, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed files:"
    for f in "${FAILED_FILES[@]}"; do echo "    $f"; done
fi
echo "══════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]
