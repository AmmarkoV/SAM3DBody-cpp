#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_model.sh — fetch SAM3DBody model files individually from HuggingFace.
#
# Replaces the "download one 5 GB zip and extract it" flow with per-file
# fetches, so a run only pulls the models its execution provider actually
# needs.  The profiles mirror the swaps resolve_backbone_defaults()
# (src/cli_common.h) performs at startup:
#
#   shared  always needed                                      ~151 MB
#   cuda    bf16 backbone + bf16 decoder (ORT CUDA EP)        ~5.1 GB
#   cpu     fp32 backbone + fp16 decoder (CPU EP has no bf16) ~3.5 GB
#   trt     fp16 TRT backbone + fp16 decoder (TensorRT EP)    ~1.8 GB
#
# Called by hand, from scripts/setup.sh, and lazily at runtime by
# ensure_models() when a binary starts up with models missing.
#
# Usage:
#   tools/fetch_model.sh [PROFILE...] [options]
#
#   PROFILE    one or more of: shared cpu cuda trt all   (default: shared cuda)
#              'shared' is implied by every other profile.
#
# Options:
#   --onnx-dir DIR   destination (default: <repo>/onnx)
#   --revision REV   HuggingFace revision (default: $SAM3D_HF_REVISION or main)
#   --list           print what would be fetched, then exit
#   --force          re-download even if the file is already present
#   -y, --yes        don't prompt for confirmation
#   -h, --help       this message
#
# Environment:
#   SAM3D_AUTO_FETCH=0   never download (refuse and print instructions)
#   SAM3D_AUTO_FETCH=1   download without prompting
#   SAM3D_HF_REVISION    default revision
#   HF_TOKEN             sent as a bearer token when set (raises rate limits)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_REPO="AmmarkoV/SAM3DBody-cpp-onnx-models"
# NOTE: pin this to a commit sha once the loose files have settled — with 'main'
# a future re-upload silently changes what an old checkout fetches.
HF_REVISION="${SAM3D_HF_REVISION:-main}"

ONNX_DIR="${REPO_ROOT}/onnx"
PROFILES=()
LIST_ONLY=0
FORCE=0
ASSUME_YES=0

# ── Manifest: profile|filename|size_bytes|sha256 ─────────────────────────────
# Sizes and hashes are the authoritative local copies; verified after download
# so a truncated .data sidecar fails loudly here instead of surfacing as a
# cryptic ONNX Runtime protobuf error at session-create time.
MANIFEST=(
  "shared|yolo.onnx|83974039|3a915ca39bf94c2f7368df028bdaea725dd619f4dc4777954617add106953833"
  "shared|pipeline.gguf|5275040|83a7ae4b7898ed4d6162a516f71a9b1ea97e0747b0f5d6902f8f0b19f3d471a2"
  "shared|body_model.lbs|27639256|9a1a3c27bc22a10d4b1558929c5138cdd0211bd70d933e560e391081ef65920c"
  "shared|correctives.bin|34272396|1c538f8a26b73f11c6de3498b2b379ca7c89dfda5d7c79b48df92047638fa22b"
  "shared|keypoint_mapping.bin|6552|8557544394112fec275c8342e2049ded8667a143a04c78fc7231b8bd0f9d559f"

  "cuda|backbone.onnx|1683984524|8d438adedf91621be019d91b8f105a6eb3e3fd7dd4c51fda2ebdaf26a0737557"
  "cuda|backbone.onnx.data|3366780928|0cdea908a53851e86fd7a9714d29dae92455cf30c0d83fe97aa6358e97ed6468"
  "cuda|decoder.onnx|97360225|7205833bf26b4dc9b877a0113495a9e0178a9f889fdf3ac785a0a5af5cc6f00f"

  "cpu|backbone_fp32.onnx|722275|9be9d7247020f63cf038b3c1e5b87ba5d75c40b3ab1dc328244a7a0540e51f5b"
  "cpu|backbone_fp32.onnx.data|3362053184|f1ba0ccc5e0b4bb9a1e61521aa66e9be659dfef524630f5935cc2c6d899d1f9f"
  "cpu|decoder_fp16.onnx|275064|c37e6b7c7328c89d69e67900d734a98ec600ff66c5f0757a434a127b646bf167"
  "cpu|decoder_fp16.onnx.data|97099804|ceee2cc6313dac633858b4580f072b431d196efcdd6e3edf1e795bbe9eab9418"

  "trt|backbone_fp16_trt.onnx|692176|edf1b67109a638b11313e8796cf0b28dac855902b29290f4e0005981571713eb"
  "trt|backbone_fp16_trt.onnx.data|1683328000|246b48f6febc86fe32d9c15410dd1a284cf3975f17d444b51acb4bd2ae94f3e7"
  "trt|decoder_fp16.onnx|275064|c37e6b7c7328c89d69e67900d734a98ec600ff66c5f0757a434a127b646bf167"
  "trt|decoder_fp16.onnx.data|97099804|ceee2cc6313dac633858b4580f072b431d196efcdd6e3edf1e795bbe9eab9418"
)

# Print the header comment block (everything after the shebang up to the first
# line that isn't a comment).
usage() {
    awk 'NR==1 {next} /^#/ {sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

# ── Args ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        shared|cpu|cuda|trt|all) PROFILES+=("$1"); shift ;;
        --onnx-dir) ONNX_DIR="$2"; shift 2 ;;
        --revision) HF_REVISION="$2"; shift 2 ;;
        --list)     LIST_ONLY=1; shift ;;
        --force)    FORCE=1; shift ;;
        -y|--yes)   ASSUME_YES=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "fetch_model.sh: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

[[ ${#PROFILES[@]} -eq 0 ]] && PROFILES=(shared cuda)
for p in "${PROFILES[@]}"; do
    [[ "$p" == "all" ]] && PROFILES=(shared cpu cuda trt) && break
done
# Every profile rests on the shared files.
PROFILES+=(shared)

_wanted_profile() {
    local want="$1"
    for p in "${PROFILES[@]}"; do [[ "$p" == "$want" ]] && return 0; done
    return 1
}

_human() {  # bytes → human readable
    awk -v b="$1" 'BEGIN{
        split("B KB MB GB TB",u," "); i=1
        while (b>=1024 && i<5) { b/=1024; i++ }
        printf (i==1 ? "%d %s" : "%.1f %s"), b, u[i]
    }'
}

# ── Work out what is actually missing ───────────────────────────────────────
# Deduplicated (decoder_fp16 appears in both the cpu and trt profiles).
NEED_NAMES=(); NEED_SIZES=(); NEED_SHAS=(); TOTAL=0

_already_queued() {
    local n="$1"
    for q in ${NEED_NAMES[@]+"${NEED_NAMES[@]}"}; do [[ "$q" == "$n" ]] && return 0; done
    return 1
}

for entry in "${MANIFEST[@]}"; do
    IFS='|' read -r prof name size sha <<< "$entry"
    _wanted_profile "$prof" || continue
    _already_queued "$name" && continue

    if [[ "$FORCE" -eq 0 && -f "${ONNX_DIR}/${name}" ]]; then
        # Size is the cheap integrity check that catches the failure mode that
        # actually happens (a truncated download).  Full sha256 over 3.4 GB on
        # every startup would cost more than it buys, so it is only verified
        # for files this script downloads itself.
        actual=$(stat -c '%s' "${ONNX_DIR}/${name}" 2>/dev/null || echo 0)
        if [[ "$actual" == "$size" ]]; then
            continue
        fi
        echo "  ${name}: size mismatch (have $(_human "$actual"), want $(_human "$size")) — refetching."
    fi

    NEED_NAMES+=("$name"); NEED_SIZES+=("$size"); NEED_SHAS+=("$sha")
    TOTAL=$(( TOTAL + size ))
done

if [[ ${#NEED_NAMES[@]} -eq 0 ]]; then
    echo "fetch_model.sh: all models for profile(s) '${PROFILES[*]}' present in ${ONNX_DIR}."
    exit 0
fi

echo "fetch_model.sh: ${#NEED_NAMES[@]} file(s), $(_human "$TOTAL") from ${HF_REPO}@${HF_REVISION}"
for i in "${!NEED_NAMES[@]}"; do
    printf '    %-34s %10s\n' "${NEED_NAMES[$i]}" "$(_human "${NEED_SIZES[$i]}")"
done

[[ "$LIST_ONLY" -eq 1 ]] && exit 0

# ── Consent ─────────────────────────────────────────────────────────────────
# A multi-GB download must never be a silent side effect of running a binary.
if [[ "${SAM3D_AUTO_FETCH:-}" == "0" ]]; then
    echo "  SAM3D_AUTO_FETCH=0 — refusing to download. Run this script by hand to fetch." >&2
    exit 1
fi
if [[ "$ASSUME_YES" -eq 0 && "${SAM3D_AUTO_FETCH:-}" != "1" ]]; then
    if [[ -t 0 ]]; then
        read -r -p "  Download $(_human "$TOTAL") to ${ONNX_DIR}? [y/N] " reply
        [[ "$reply" =~ ^[Yy] ]] || { echo "  Aborted."; exit 1; }
    else
        echo "  Not a terminal and no --yes / SAM3D_AUTO_FETCH=1 — refusing to download." >&2
        echo "  Fetch them first with:  bash tools/fetch_model.sh ${PROFILES[*]} --yes" >&2
        exit 1
    fi
fi

command -v curl >/dev/null 2>&1 || {
    echo "fetch_model.sh: curl not found on PATH — install it and retry." >&2; exit 1; }

mkdir -p "${ONNX_DIR}"

# ── Fetch ───────────────────────────────────────────────────────────────────
CURL_AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")

for i in "${!NEED_NAMES[@]}"; do
    name="${NEED_NAMES[$i]}"; size="${NEED_SIZES[$i]}"; sha="${NEED_SHAS[$i]}"
    url="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/${name}?download=true"
    dest="${ONNX_DIR}/${name}"
    part="${dest}.part"

    echo "  [$((i+1))/${#NEED_NAMES[@]}] ${name}  ($(_human "$size"))"

    # Resume only when a partial exists and is shorter than the target; a .part
    # already at full size would make curl request an unsatisfiable range (416)
    # and --fail would turn that into an error.
    have=$( [[ -f "$part" ]] && stat -c '%s' "$part" 2>/dev/null || echo 0 )
    if [[ "$have" -lt "$size" ]]; then
        resume=()
        [[ "$have" -gt 0 ]] && resume=(--continue-at -)
        curl -L --fail --progress-bar --retry 3 --retry-delay 2 \
             "${CURL_AUTH[@]}" ${resume[@]+"${resume[@]}"} -o "$part" "$url"
    else
        echo "        complete partial found — verifying."
    fi

    actual=$(stat -c '%s' "$part" 2>/dev/null || echo 0)
    if [[ "$actual" != "$size" ]]; then
        echo "  ERROR: ${name} is $(_human "$actual"), expected $(_human "$size")." >&2
        echo "         Partial kept at ${part} — rerun to resume." >&2
        exit 1
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$part" | cut -d' ' -f1)
        if [[ "$got" != "$sha" ]]; then
            echo "  ERROR: ${name} sha256 mismatch." >&2
            echo "         expected ${sha}" >&2
            echo "         got      ${got}" >&2
            rm -f "$part"
            exit 1
        fi
    fi

    # Atomic: the final name never exists in a half-written state, so the
    # ifstream(...).good() probes in cli_common.h can trust it.
    mv -f "$part" "$dest"
done

echo "fetch_model.sh: done — ${#NEED_NAMES[@]} file(s) in ${ONNX_DIR}."
