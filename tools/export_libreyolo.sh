#!/usr/bin/env bash
# tools/export_libreyolo.sh
#
# Venv wrapper for tools/export_libreyolo.py.
#
# Creates a local venv (tools/.venv) on first run, installs the required
# packages, then runs export_libreyolo.py with all arguments forwarded.
#
# Usage:
#   tools/export_libreyolo.sh --weights LibreYOLO9t.pt --out ./onnx/libreyolo9.onnx
#   tools/export_libreyolo.sh --weights LibreYOLO9t.pt --out ./onnx/libreyolo9.onnx --imgsz 640
#
# All flags are forwarded verbatim to export_libreyolo.py; run with --help for
# the full option list. The script prints the resulting ONNX output shape —
# confirm it is [1, 84, N] before using `--detector libreyolo`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
REQS="libreyolo onnx onnxruntime numpy"

# ── Bootstrap venv on first run ───────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python3" ]]; then
    echo "[export_libreyolo] Creating venv at $VENV …"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    echo "[export_libreyolo] Installing: $REQS"
    "$VENV/bin/pip" install --quiet $REQS
    echo "[export_libreyolo] Venv ready."
    echo
fi

# ── Run ───────────────────────────────────────────────────────────────────────
exec "$VENV/bin/python3" "$SCRIPT_DIR/export_libreyolo.py" "$@"
