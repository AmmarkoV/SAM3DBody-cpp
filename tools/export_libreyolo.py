#!/usr/bin/env python3
"""
tools/export_libreyolo.py  —  export a LibreYOLO detector to ONNX

LibreYOLO (https://github.com/LibreYOLO/libreyolo) is a Python library; the
SAM-3D-Body C++ pipeline cannot link it directly. This script exports a
LibreYOLO model to ONNX so it can be loaded through the existing ONNX Runtime
detector path and selected at runtime with `--detector libreyolo`.

The YOLOv9 detection family exports an output tensor of shape [1, 84, N]
(4 bbox + 80 COCO class scores; person = class 0). The C++ side parses this
bbox-only — keypoints are not produced by detection models.

Usage
-----
  tools/export_libreyolo.sh --weights LibreYOLO9t.pt --out ./onnx/libreyolo9.onnx
  tools/export_libreyolo.sh --weights LibreYOLO9t.pt --out ./onnx/libreyolo9.onnx --imgsz 640

All flags are forwarded by the .sh wrapper; run with --help for the option list.
"""
import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description="Export a LibreYOLO model to ONNX")
    ap.add_argument("--weights", default="LibreYOLO9t.pt",
                    help="LibreYOLO weights (.pt). Downloaded from HF on first use "
                         "if not a local path. Default: LibreYOLO9t.pt")
    ap.add_argument("--out", default="./onnx/libreyolo9.onnx",
                    help="Output ONNX path (default: ./onnx/libreyolo9.onnx)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Inference input size; must match the C++ letterbox (640)")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset (default 12)")
    args = ap.parse_args()

    try:
        from libreyolo import LibreYOLO
    except ImportError:
        sys.exit("[export_libreyolo] libreyolo is not installed. Run via "
                 "tools/export_libreyolo.sh, which bootstraps a venv.")

    print(f"[export_libreyolo] Loading {args.weights} …")
    model = LibreYOLO(args.weights)

    print(f"[export_libreyolo] Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}) …")
    # LibreYOLO follows the Ultralytics-style export API and returns the path to
    # the produced .onnx file.
    produced = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)

    # `export` may return a path string or a list; normalise.
    if isinstance(produced, (list, tuple)):
        produced = produced[0]
    produced = str(produced)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(produced) != os.path.abspath(args.out):
        shutil.move(produced, args.out)
    print(f"[export_libreyolo] Wrote {args.out}")

    # Report the output tensor shape so the user can confirm it is [1, 84, N]
    # before wiring `--detector libreyolo`.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        for o in sess.get_outputs():
            print(f"[export_libreyolo] output '{o.name}' shape={o.shape}")
        print("[export_libreyolo] Expect a feature dim of 84 (4 bbox + 80 classes); "
              "person is class 0. Then run with: "
              f"--yolo {args.out} --detector libreyolo")
    except Exception as e:  # noqa: BLE001 — shape check is best-effort
        print(f"[export_libreyolo] (could not introspect output shape: {e})")


if __name__ == "__main__":
    main()
