#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_model.sh — fetch SAM3DBody model files individually from HuggingFace.
#
# Replaces the "download one 5 GB zip and extract it" flow with per-file
# fetches, so a run only pulls the models its execution provider actually
# needs.  The profiles mirror the swaps resolve_backbone_defaults()
# (src/cli_common.h) performs at startup:
#
#   shared   always needed                                      ~151 MB
#   cuda     bf16 backbone + bf16 decoder (ORT CUDA EP)        ~5.1 GB
#   cpu      fp32 backbone + fp16 decoder (CPU EP has no bf16) ~3.5 GB
#   trt      fp16 TRT backbone + fp16 decoder (TensorRT EP)    ~1.8 GB
#   refined  extra files for --refined-pose (see PLAN.md,
#            issue #15 "refined pose" plan) — opt-in, not part
#            of 'all'; combine with a base profile, e.g.
#            'tools/fetch_model.sh cuda refined'              ~607 MB
#
# Called by hand, from scripts/setup.sh, and lazily at runtime by
# ensure_models() when a binary starts up with models missing.
#
# Usage:
#   tools/fetch_model.sh [PROFILE...] [options]
#
#   PROFILE    one or more of: shared cpu cuda trt refined all
#              (default: shared cuda). 'shared' is implied by every
#              other profile. 'all' does NOT include 'refined' — it
#              stays opt-in since most users don't need it.
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

  # --refined-pose (see PLAN.md, issue #15 "refined pose" plan). Opt-in —
  # not part of 'all'. pipeline.gguf's own entry above is untouched by
  # this: the hand-decoder heads live in a SEPARATE pipeline_refined.gguf
  # specifically so that entry never has to change for non-refined users.
  #
  # The 28 decoder_* ONNX graphs are the iterative per-layer exports driven
  # by the C++ pass-1/pass-2/hand-crop refinement loops (9 decoder_hand,
  # 9 decoder_prompted, 10 decoder_pass1 incl. the hand-box head) — the
  # earlier single-shot decoder_hand.onnx / decoder_prompted.onnx /
  # decoder_handbox_fp32.onnx are no longer loaded and were removed from
  # this manifest.
  "refined|decoder_hand_pre.onnx|22468206|c1fd55aca1f81d2fd49cc62404bf1f8c0915bc5f6339b32823a05dbdbd0a7383"
  "refined|decoder_hand_layer0.onnx|26313271|96158003c438be953e09fb72c8657069fd32a5f893efba1113a47a5a7d84a938"
  "refined|decoder_hand_layer1.onnx|26313271|cdd05ce2116ae2711b9f02f78a2cdb4c32dfa5a1456c3f9f3c1726b63e041336"
  "refined|decoder_hand_layer2.onnx|26313271|c5ddf7c21bfd4125820158ad99d41b00fa4df2f59a0502f178bf66e545f46f11"
  "refined|decoder_hand_layer3.onnx|26313271|a59d0bca06f96921af5175058f078809b197ee4680fb0a592c89ea099df76181"
  "refined|decoder_hand_layer4.onnx|26313271|d6761be5667baaf8d82d0e8a0ebe137b6829a8093e1499a7462f51dd254adfd5"
  "refined|decoder_hand_layer5.onnx|26313271|55ca5380f57cac98b5d1761fc19eaa3967c62f74d54b381ef6379f306130acbf"
  "refined|decoder_hand_normfinal.onnx|8739|5cc6b57b5ba06ddddd208620f0ed6715376615e22fe67db7b60acd8a17856529"
  "refined|decoder_hand_update.onnx|13700938|8e45a18bb37b93506ec2664552f4d04f251a9177a6c2c928b31fa9c12fb9e7cf"
  "refined|decoder_prompted_pre.onnx|22839099|feb0ff78efcf2b99b2fd17cc39d6f060d059ce43e03c6c03e5e228466418384e"
  "refined|decoder_prompted_layer0.onnx|26313271|f63530c8efcca70872fcc8f086b9e3af202c29f7978052d1ebc82976cbec31ed"
  "refined|decoder_prompted_layer1.onnx|26313271|e4d8de10105d024aefd176e0e176b3cf344dc5fc99b82fa311889671314c4c3d"
  "refined|decoder_prompted_layer2.onnx|26313271|b1b153db078c35352252093ad0616672d3fa5d1698b0aab28df485ce87d36a67"
  "refined|decoder_prompted_layer3.onnx|26313271|63d1b61ed26485c5f6689ca0f1ba8a28ce37c8ac0f58a7a9944fa5e686c715ae"
  "refined|decoder_prompted_layer4.onnx|26313271|33fd82d4b6bc4ca911ad9b1e278750885f05cb2a1a07fde571c2f1b7ad621f1d"
  "refined|decoder_prompted_layer5.onnx|26313271|5d756b0dc7328cc42d727e4b40b735f7c74c831dce72f461594f3c5e8898cff5"
  "refined|decoder_prompted_normfinal.onnx|8739|9464b34a1c2018194415fcd3799c21ed14d9ce9900fcb0124b24fcce912f95ba"
  "refined|decoder_prompted_update.onnx|13700977|c0332d2bf148a339acc401ce1c87aff9235362472234a951b94e3655e9bd681b"
  "refined|decoder_pass1_pre.onnx|22468556|6bfec00c648fd0a52e09987995140b7f9b09fbd68922eccd8f54dec759644bf4"
  "refined|decoder_pass1_layer0.onnx|26313271|15228ab564fecb288f4724da79c29847972c48ac78e8e08aa97b9e891d1edd97"
  "refined|decoder_pass1_layer1.onnx|26313271|f0950409d012c4ff4d77a321bc5e59f1f06815c1e3b02e5d57fc3ba54e13f4ec"
  "refined|decoder_pass1_layer2.onnx|26313271|e70e96579cf74687cc33973d8c3513ecf9befa03ac05f1e7ce54d1b72e1c7fe3"
  "refined|decoder_pass1_layer3.onnx|26313271|24e755a100c5f4ea3e480b75300eff3533bce8a0959dd5f9864f13145551f76d"
  "refined|decoder_pass1_layer4.onnx|26313271|880d7bfd80f8292727c30a804283d17dab1edabd8d650394a81c39f80eb047d4"
  "refined|decoder_pass1_layer5.onnx|26313271|2485df418ec1bb8cfff05e968405f1458718681b292c432b0d5e398ec1febcf5"
  "refined|decoder_pass1_normfinal.onnx|8739|ba75620503fe950a2c85e4042183119e9ba9c6c3aa2565c46650dcf43f862515"
  "refined|decoder_pass1_update.onnx|13700938|b0fa19f7455ac0b15630617ed608396d6c8995adb86c1da4e7efec8dedf53a7f"
  "refined|decoder_pass1_handbox.onnx|8432332|f481ef0fdbc1f0b2da44b197ebbdb746739a64cd998b97cca3a59786e0a60924"
  "refined|pipeline_refined.gguf|14764576|88ec6f7bdbf8519f5016c87cbb9c72b4db50e0ea2d71f298593f00a8999b9951"
)

# Print the header comment block (everything after the shebang up to the first
# line that isn't a comment).
usage() {
    awk 'NR==1 {next} /^#/ {sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

# ── Args ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        shared|cpu|cuda|trt|refined|all) PROFILES+=("$1"); shift ;;
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
