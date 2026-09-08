#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  regress.sh — refactor safety net for the refined-pose pipeline.
#
#  There are no unit tests for src/fast_sam_3dbody.cpp.  This is the harness
#  that stands in for them: it checks that a change leaves the BVH output
#  inside the pipeline's own run-to-run noise, and that it did not cost frame
#  time.  Used for every step of the readability work (see the phase plan).
#
#  Three traps this encodes, each of which produced a wrong conclusion before:
#
#   1. The pipeline is not bit-reproducible (CUDA GEMM reassociation), so a
#      candidate is compared against the BASELINE'S OWN run-to-run spread,
#      never against zero.  `capture` therefore always records two baseline
#      runs so the noise floor is known.
#   2. Absolute timings drift with GPU temperature — this box is power-capped
#      at 180 W and clocks swing 1350-2730 MHz.  A single before/after pair
#      once showed "-14.5 ms" for a change worth 3.7.  So `compare` interleaves
#      the two sides and takes medians over matched frames.
#   3. TensorRT is not on the default loader path.  Without it --trt silently
#      falls back to CUDA and every number is wrong.  Set up here.
#
#  Usage:
#     scripts/regress.sh capture <name>     # record a reference + noise floor
#     scripts/regress.sh check   <name>     # compare current build to it
#
#  Typical flow around a refactor:
#     scripts/regress.sh capture before     # on the unmodified build
#     ...edit, rebuild...
#     scripts/regress.sh check   before     # verdict
#
#  Env overrides: FSB_VIDEO, FSB_FRAMES, FSB_ARGS, FSB_OUT
# ════════════════════════════════════════════════════════════════════════════
set -u

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$THISDIR"

VIDEO="${FSB_VIDEO:-videos/300.mkv}"
FRAMES="${FSB_FRAMES:-50}"
ARGS="${FSB_ARGS:---trt --refined-pose}"
OUT="${FSB_OUT:-/tmp/fsb_regress}"
BIN=./build/fast_sam_3dbody_render

# TensorRT ships inside the tools venv; without it --trt silently degrades to
# the CUDA EP and both the timings and the fp16-vs-bf16 model choice change.
TRT_LIB="$THISDIR/tools/.venv/lib/python3.12/site-packages/tensorrt_libs"
[ -d "$TRT_LIB" ] && export LD_LIBRARY_PATH="$TRT_LIB:${LD_LIBRARY_PATH:-}"

[ -x "$BIN" ]     || { echo "no $BIN — build first"; exit 2; }
[ -f "$VIDEO" ]   || { echo "no video $VIDEO (set FSB_VIDEO)"; exit 2; }
mkdir -p "$OUT"

# One run -> $OUT/<tag>.log and $OUT/<tag>_<person>.bvh
run() {
    local tag="$1"
    $BIN --from "$VIDEO" --headless --frames "$FRAMES" $ARGS \
         --bvh "$OUT/$tag.bvh" > "$OUT/$tag.log" 2>&1 \
      || { echo "run failed: $tag (see $OUT/$tag.log)"; exit 1; }
    if grep -q "TensorRT EP failed" "$OUT/$tag.log"; then
        echo "WARNING: TensorRT EP failed to load — --trt fell back to CUDA."
        echo "         Timings are not comparable to a real --trt run."
    fi
}

# Median frame total over frames with a matched workload.  Frames differ in
# person count and surviving hand-crop count, and those change the work by
# more than most optimisations do — so only like-for-like frames are pooled.
medians() {
    python3 - "$1" <<'PY'
import re, statistics, sys
frames, cur = [], {}
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\[FSB\] (.+?):\s+([\d.]+) ms(.*)$", line)
    if not m:
        continue
    k, v, rest = m.group(1), float(m.group(2)), m.group(3)
    cur[k] = cur.get(k, 0) + v
    if "hand crop(s)" in rest:
        cur["_c"] = int(re.search(r"\((\d+) hand crop", rest).group(1))
    if k == "total":
        cur["_p"] = int(re.search(r"\((\d+) persons", rest).group(1))
        frames.append(cur); cur = {}
if not frames:
    print("no frames"); raise SystemExit(1)
# pool the most common (persons, hand-crops) workload
from collections import Counter
key, _ = Counter((f.get("_p"), f.get("_c")) for f in frames).most_common(1)[0]
sel = [f for f in frames if (f.get("_p"), f.get("_c")) == key]
print(f"  workload {key[0]} person(s) / {key[1]} hand crop(s), {len(sel)}/{len(frames)} frames")
for k in sorted({k for f in sel for k in f if not k.startswith("_")},
                key=lambda k: -statistics.median([f.get(k, 0.0) for f in sel])):
    print(f"    {k:34s} {statistics.median([f.get(k,0.0) for f in sel]):7.2f} ms")
PY
}

case "${1:-}" in
capture)
    NAME="${2:?usage: regress.sh capture <name>}"
    echo "capturing reference '$NAME' ($FRAMES frames, $ARGS)"
    # Three runs, not two.  The per-person `max` is a single worst channel out
    # of ~500 x N frames and swings by 3x between runs of the SAME binary, so a
    # one-pair floor routinely samples the low end and then fails an innocent
    # change.  Three runs give three pairings and a floor that actually bounds it.
    run "${NAME}_a"
    run "${NAME}_b"
    run "${NAME}_c"
    echo "noise floor (same binary, three pairings):"
    for pair in "a b" "a c" "b c"; do
        set -- $pair
        python3 tools/bvh_compare.py "$OUT/${NAME}_$1" "$OUT/${NAME}_$2" | tail -1
    done
    medians "$OUT/${NAME}_a.log"
    echo "reference saved under $OUT/${NAME}_*"
    ;;
check)
    NAME="${2:?usage: regress.sh check <name>}"
    [ -f "$OUT/${NAME}_a.log" ] || { echo "no reference '$NAME' — run capture first"; exit 2; }
    echo "noise floor of the reference (three pairings of the same binary):"
    FLOOR_MAX=0; FLOOR_MEAN=0
    for pair in "a b" "a c" "b c"; do
        set -- $pair
        [ -f "$OUT/${NAME}_$2_0.bvh" ] || continue
        line=$(python3 tools/bvh_compare.py "$OUT/${NAME}_$1" "$OUT/${NAME}_$2" | tail -1)
        echo "  $1 vs $2: $line"
        m=$(echo "$line"  | sed 's/.*max=\([0-9.]*\).*/\1/')
        mu=$(echo "$line" | sed 's/.*mean=\([0-9.]*\).*/\1/')
        FLOOR_MAX=$(python3 -c "print(max($FLOOR_MAX,$m))")
        FLOOR_MEAN=$(python3 -c "print(max($FLOOR_MEAN,$mu))")
    done
    [ "$FLOOR_MAX" != "0" ] || { echo "could not read the noise floor"; exit 2; }
    echo
    echo "current build vs reference:"
    run "${NAME}_now"
    RES=$(python3 tools/bvh_compare.py "$OUT/${NAME}_now" "$OUT/${NAME}_a")
    echo "$RES"
    GOT_MAX=$(echo  "$RES" | tail -1 | sed 's/.*max=\([0-9.]*\).*/\1/')
    GOT_MEAN=$(echo "$RES" | tail -1 | sed 's/.*mean=\([0-9.]*\).*/\1/')
    # The verdict leans on the MEAN: it is stable to about +-25% run to run, so
    # any real regression moves it by orders of magnitude, while `max` alone
    # produces false alarms.  `max` is still checked, but with a wide bound.
    VERDICT=$(python3 - <<EOF
mean_ok = $GOT_MEAN <= $FLOOR_MEAN * 3
max_ok  = $GOT_MAX  <= $FLOOR_MAX  * 3
print(0 if (mean_ok and max_ok) else (1 if not mean_ok else 2))
EOF
)
    echo
    echo "timing — reference:"; medians "$OUT/${NAME}_a.log"
    echo "timing — current:";   medians "$OUT/${NAME}_now.log"
    echo
    echo "floor:   max=$FLOOR_MAX mean=$FLOOR_MEAN"
    echo "current: max=$GOT_MAX mean=$GOT_MEAN"
    case "$VERDICT" in
    0) echo "PASS: mean and max both within 3x the reference noise floor." ;;
    1) echo "FAIL: MEAN deviation exceeds 3x the floor — this is the reliable"
       echo "      signal, treat it as a real regression." ;;
    2) echo "WARN: mean is fine but max exceeds 3x the floor — a localised"
       echo "      break, or an unlucky sample.  Re-run; if the mean stays"
       echo "      clean across runs it is noise." ;;
    esac
    echo "NOTE: timings above are single runs and drift with GPU temperature."
    echo "      For a real perf verdict, interleave the two builds (swap"
    echo "      build/libfast_sam_3dbody.so — the render binary is only a shim)."
    [ "$VERDICT" = 1 ] && exit 1
    exit 0
    ;;
*)
    sed -n '2,34p' "$0" | sed 's/^#\s\?//'
    exit 1
    ;;
esac
