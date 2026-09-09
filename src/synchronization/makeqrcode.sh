#!/usr/bin/env bash
set -euo pipefail

# Output folder for frames
OUT_DIR="frames"
mkdir -p "$OUT_DIR"

# Start time: now (UTC by default here). You can change this to local time if you want.
# We'll generate 60 timestamps: start + 0..59 seconds.
START_EPOCH="$(date -u +%s)"

# Customize what gets encoded in the QR.
# Example payload: unix epoch + ISO8601 + frame index
# You can change the format freely (JSON is also nice).
for i in $(seq 0 59); do
  ts=$((START_EPOCH + i))
  iso="$(date -u -d "@$ts" +"%Y-%m-%dT%H:%M:%SZ")"

  payload="ts_unix=$ts&ts_iso=$iso&frame=$i"

  # Zero-padded filename
  fname="$(printf "image_%06d.png" "$i")"
  outpath="$OUT_DIR/$fname"

  # -s = module size in pixels, -m = margin modules
  qrencode -o "$outpath" -s 10 -m 2 "$payload"
done

echo "Done. Generated 60 frames in: $OUT_DIR/"

