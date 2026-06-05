#!/usr/bin/env python3
"""
tools/export_backbone_fp16.py  —  run-once FP16 conversion for backbone.onnx

WHY THIS EXISTS
---------------
The backbone is a DINOv3-ViT-H (~630 M params) exported in float32
(backbone.onnx + backbone.onnx.data, ~4.8 GB).  ONNX Runtime's CUDA EP runs a
model in its *native* precision — there is no runtime "use fp16" switch on the
CUDA EP (only the TensorRT EP has trt_fp16_enable).  So the FP32 backbone runs
FP32 on the GPU and never touches the Ada/Turing tensor cores.

Converting the weights to float16 lets the CUDA EP issue half-precision GEMMs on
the tensor cores.  On the tested RTX 1000 Ada laptop GPU this roughly halves the
backbone time (~240 ms → ~120 ms) and the on-disk / VRAM footprint:

    backbone.onnx + .data        ~4.8 GB  FP32
    backbone_fp16.onnx + .data   ~2.4 GB  FP16

I/O TYPES
---------
By default the converted model KEEPS its graph inputs and outputs in float32
(keep_io_types=True): the model casts FP32→FP16 on entry and FP16→FP32 on exit
internally.  This means the C++ side feeds and reads the exact same FP32 tensors
as before — no code change is required to *use* the model, only to *load* it.
(The loader auto-prefers backbone_fp16.onnx on CUDA when present; see
src/cli_common.h resolve_backbone_defaults.)

NUMERICS
--------
A handful of ops are numerically sensitive in half precision.  The converter's
default op_block_list keeps the worst offenders (e.g. some normalisations) in
FP32, and --min-positive-val / --max-finite-val clamp tiny/huge constants so they
don't underflow/overflow to 0/inf when narrowed.  DINOv3 was trained in
bf16/fp16-friendly ranges, so FP16 inference is visually indistinguishable from
FP32 here; if you ever see NaNs, add the offending op type via --block-op.

USAGE
-----
    # Run-once conversion (needs ~6–8 GB free RAM for the 4.8 GB model):
    python3 tools/export_backbone_fp16.py --onnx-dir ./onnx

    # Then just run normally — on CUDA the loader picks backbone_fp16.onnx up
    # automatically; or pin it explicitly:
    ./build/fast_sam_3dbody_run --from input.mp4 --backbone backbone_fp16.onnx

The original backbone.onnx is never modified — both coexist in onnx/.

DEPENDENCIES
------------
    pip3 install onnx onnxconverter-common numpy
"""

import argparse
import os
import sys
import time

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
for _imp, _pip in (("onnx", "onnx"),
                   ("onnxconverter_common", "onnxconverter-common"),
                   ("numpy", "numpy")):
    try:
        __import__(_imp)
    except ImportError:
        _missing.append(_pip)
if _missing:
    print("Missing packages:", " ".join(_missing))
    print("  pip3 install", " ".join(_missing))
    sys.exit(1)

import onnx
from onnxconverter_common import float16


def dir_size_gb(directory: str, prefix: str) -> float:
    """Sum of files in `directory` whose name is `prefix` or `prefix.*` (external data)."""
    if not os.path.isdir(directory):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f == prefix or f.startswith(prefix + ".")
    )
    return total / 1e9


def main():
    ap = argparse.ArgumentParser(
        description="Convert backbone.onnx weights to float16 for CUDA tensor cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--onnx-dir", default="./onnx",
                    help="Directory containing backbone.onnx (default: ./onnx)")
    ap.add_argument("--input", default=None,
                    help="Override input path (default: <onnx-dir>/backbone.onnx)")
    ap.add_argument("--output", default=None,
                    help="Override output path (default: <onnx-dir>/backbone_fp16.onnx)")
    ap.add_argument("--cast-io", action="store_true",
                    help="Also narrow graph inputs/outputs to float16 (keep_io_types=False). "
                         "Requires matching C++ changes; NOT recommended — leave I/O as fp32.")
    ap.add_argument("--block-op", action="append", default=[], metavar="OPTYPE",
                    help="Keep this op type in FP32 (repeatable). Added to the default block list.")
    ap.add_argument("--min-positive-val", type=float, default=1e-7,
                    help="Smallest positive constant kept before flush-to-zero (default 1e-7)")
    ap.add_argument("--max-finite-val", type=float, default=1e4,
                    help="Largest finite constant before clamping (default 1e4)")
    args = ap.parse_args()

    in_path = args.input or os.path.join(args.onnx_dir, "backbone.onnx")
    out_path = args.output or os.path.join(args.onnx_dir, "backbone_fp16.onnx")

    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found")
        sys.exit(1)

    in_dir = os.path.dirname(os.path.abspath(in_path))
    in_base = os.path.basename(in_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    out_base = os.path.basename(out_path)
    in_gb = dir_size_gb(in_dir, in_base)

    print(f"Input:   {in_path}  ({in_gb:.2f} GB)")
    print(f"Output:  {out_path}")
    print(f"I/O:     {'float16 (cast-io)' if args.cast_io else 'float32 (kept; no C++ change needed)'}")
    print(f"onnx:    {onnx.__version__}")

    # The 4.8 GB FP32 model is held in RAM during conversion.  Warn if tight.
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < in_gb * 1.6:
            print(f"\nWARNING: {avail_gb:.1f} GB RAM available, model is {in_gb:.2f} GB.")
            print("         FP16 conversion needs ~1.5–2× the model size in RAM.")
    except ImportError:
        pass

    print("\nLoading FP32 model (with external data) …")
    t0 = time.time()
    model = onnx.load(in_path, load_external_data=True)

    # Default block list keeps fp16-unstable ops in fp32; extend with --block-op.
    op_block_list = list(float16.DEFAULT_OP_BLOCK_LIST) + list(args.block_op)
    if args.block_op:
        print(f"  Extra FP32-kept ops: {args.block_op}")

    print("Converting weights/activations to float16 …")
    # disable_shape_infer=True: ONNX shape inference can't run on a >2 GB model,
    # and the conversion doesn't need it here.
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=not args.cast_io,
        op_block_list=op_block_list,
        min_positive_val=args.min_positive_val,
        max_finite_val=args.max_finite_val,
        disable_shape_infer=True,
    )

    # The fp16 model is still > 2 GB → must serialise weights as external data
    # (protobuf single-file hard limit is 2 GB).  One sidecar: <out_base>.data.
    print(f"Saving {out_path} (external data → {out_base}.data) …")
    # Remove any stale sidecar so save() starts clean.
    stale = os.path.join(out_dir, out_base + ".data")
    if os.path.exists(stale):
        os.remove(stale)
    onnx.save(
        model_fp16,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_base + ".data",
        convert_attribute=True,
    )

    elapsed = time.time() - t0
    out_gb = dir_size_gb(out_dir, out_base)
    ratio = in_gb / out_gb if out_gb > 0 else float("nan")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Output size: {out_gb:.2f} GB  (was {in_gb:.2f} GB, {ratio:.1f}× smaller)")
    print()
    print("Original backbone.onnx is untouched — both models coexist in the onnx/ dir.")
    print()
    print("On CUDA the loader auto-prefers backbone_fp16.onnx; or pin it:")
    print(f"  --backbone {out_base}")
    print()
    print("Compare against FP32:")
    print(f"  FP32:  ./build/fast_sam_3dbody_run --from input.mp4 --backbone {in_base}")
    print(f"  FP16:  ./build/fast_sam_3dbody_run --from input.mp4 --backbone {out_base}")


if __name__ == "__main__":
    main()
