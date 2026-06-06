#!/usr/bin/env python3
"""
tools/export_backbone_fp16.py  —  run-once bfloat16 → float16 remap for ONNX models

WHY THIS EXISTS
---------------
The backbone (DINOv3-ViT-H, ~630 M params) and the decoder are exported with
**bfloat16** weights (355 bf16 tensors in the backbone), with Cast→bf16 / Cast→
fp32 nodes threading bf16 storage through compute.  This tool remaps every bf16
tensor in the graph to **float16** (initializers, Cast targets, Constant /
ConstantOfShape attribute tensors, and value_info), so:

  * ORT's CUDA EP uses its mature fp16 kernels instead of weaker bf16 ones; and
  * the ONNX Runtime **TensorRT EP** accepts the model at all — TRT has no bf16
    path and rejects bf16 subgraph-boundary tensors at run time
    ("TensorRT EP output tensor data type: 16 not supported").

  bf16 weights, |max| ≤ 62  →  fp16 (range ±65504; safe, +mantissa precision)

WHY THE FP16 ATTRIBUTE PASS MATTERS (don't drop it):
ConstantOfShape / Constant carry their *own* dtype in a `value` attribute.
Converting initializers + value_info but missing these leaves the graph
inconsistent — a fp16 value_info over a bf16 ConstantOfShape — which ORT rejects
at load ("Type (tensor(float16)) … does not match expected type (bfloat16)").

MEASURED RESULT (RTX 1000 Ada laptop, DINOv3-ViT-H, batch=1, 25-frame steady state)
-----------------------------------------------------------------------------------
  on disk             :  5.05 GB  →  1.68 GB   (3× smaller, ~3.3 GB less VRAM)
  backbone, CUDA EP   :  ~245 ms (bf16)  →  ~230 ms (fp16)     (~6% faster)
  backbone, TRT EP    :  ~148 ms                               (1.6× vs CUDA EP)

The big win is TensorRT: the heavy GEMMs are already fp16 in the graph, but ORT's
CUDA EP schedules them far worse than a TRT fp16 engine does.  See tools/run_trt.sh
and src/cli_common.h resolve_backbone_defaults for the --trt wiring.  Note the
backbone's `If` (rope) subgraphs must additionally be folded for TRT to build it
(backbone_fp16_trt.onnx); this tool does the bf16→fp16 remap only.

Graph inputs/outputs stay float32, so the C++ side feeds/reads the same FP32
buffers — no inference-code change needed, only a load-time model swap.

USAGE
-----
    # Backbone (default paths):
    python3 tools/export_backbone_fp16.py --onnx-dir ./onnx
    # Decoder (TRT needs this; stock decoder.onnx is bf16 → TRT-incompatible):
    python3 tools/export_backbone_fp16.py --input  onnx/decoder.onnx \
                                          --output onnx/decoder_fp16.onnx

Originals are never modified — converted models coexist in onnx/.  If the input
has no bf16 tensors, the tool reports "nothing to convert" and exits.

DEPENDENCIES
------------
    pip3 install onnx numpy ml_dtypes
"""

import argparse
import os
import sys
import time

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
for _imp, _pip in (("onnx", "onnx"), ("numpy", "numpy"), ("ml_dtypes", "ml_dtypes")):
    try:
        __import__(_imp)
    except ImportError:
        _missing.append(_pip)
if _missing:
    print("Missing packages:", " ".join(_missing))
    print("  pip3 install", " ".join(_missing))
    sys.exit(1)

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

BF16 = TensorProto.BFLOAT16   # 16
F16  = TensorProto.FLOAT16    # 10


def dir_size_gb(directory: str, prefix: str) -> float:
    """Sum of files in `directory` named `prefix` or `prefix.*` (external data)."""
    if not os.path.isdir(directory):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f == prefix or f.startswith(prefix + ".")
    )
    return total / 1e9


def _remap_tensor_bf16(t, counts, key):
    """If TensorProto `t` is bf16, rewrite it to fp16 in place (bf16→fp32→fp16)."""
    if t.data_type == BF16:
        arr = numpy_helper.to_array(t).astype(np.float32).astype(np.float16)
        new_t = numpy_helper.from_array(arr, t.name)
        t.CopyFrom(new_t)
        counts[key] += 1


def remap_graph_bf16_to_fp16(graph, counts):
    """Recursively rewrite every bfloat16 in `graph` (and its subgraphs) to float16."""
    # 1. Initializers: convert bf16 weight values bf16 → fp32 → fp16, in place.
    for t in graph.initializer:
        _remap_tensor_bf16(t, counts, "init")

    for node in graph.node:
        # 2a. Cast nodes that target bf16 → target fp16.
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == BF16:
                    a.i = F16
                    counts["cast"] += 1
        # 2b. Tensor-valued attributes (Constant `value`, ConstantOfShape `value`,
        #     …): these carry their own dtype and produce a bf16 *output* the
        #     value_info pass below can't fix.  Missing these leaves the graph
        #     inconsistent — e.g. a fp16 value_info over a bf16 ConstantOfShape,
        #     which ORT rejects at load ("type … does not match expected type").
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                _remap_tensor_bf16(a.t, counts, "attr")
            elif a.type == onnx.AttributeProto.TENSORS:
                for t in a.tensors:
                    _remap_tensor_bf16(t, counts, "attr")
            # Recurse into subgraph attributes (If/Loop/Scan bodies).
            elif a.type == onnx.AttributeProto.GRAPH:
                remap_graph_bf16_to_fp16(a.g, counts)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sub in a.graphs:
                    remap_graph_bf16_to_fp16(sub, counts)

    # 3. Tensor-type annotations on value_info / inputs / outputs.  (Graph I/O are
    #    fp32 here, so this only ever touches internal bf16 value_info.)
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        if tt.elem_type == BF16:
            tt.elem_type = F16
            counts["vinfo"] += 1


def main():
    ap = argparse.ArgumentParser(
        description="Remap backbone.onnx weights from bfloat16 to float16 for CUDA tensor cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--onnx-dir", default="./onnx",
                    help="Directory containing backbone.onnx (default: ./onnx)")
    ap.add_argument("--input", default=None,
                    help="Override input path (default: <onnx-dir>/backbone.onnx)")
    ap.add_argument("--output", default=None,
                    help="Override output path (default: <onnx-dir>/backbone_fp16.onnx)")
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
    print(f"onnx:    {onnx.__version__}")

    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < in_gb * 1.6:
            print(f"\nWARNING: {avail_gb:.1f} GB RAM available, model is {in_gb:.2f} GB.")
            print("         Conversion holds the model in RAM (~1.5–2× its size).")
    except ImportError:
        pass

    print("\nLoading model (with external data) …")
    t0 = time.time()
    model = onnx.load(in_path, load_external_data=True)

    # Sanity: how much is actually bf16?
    n_bf16 = sum(1 for t in model.graph.initializer if t.data_type == BF16)
    n_fp32 = sum(1 for t in model.graph.initializer if t.data_type == TensorProto.FLOAT)
    n_fp16 = sum(1 for t in model.graph.initializer if t.data_type == F16)
    print(f"  initializers: bf16={n_bf16}  fp32={n_fp32}  fp16={n_fp16}")
    if n_bf16 == 0:
        print("\nNothing to convert: no bfloat16 tensors found.")
        print("  This model is not a bf16 export — see the NOTE in this file's header.")
        sys.exit(2)

    # Range guard: bf16 carries fp32 range; fp16 caps at ±65504.  Verify no weight
    # would overflow before we narrow (NN weights are tiny, but check to be safe).
    wmax = 0.0
    for t in model.graph.initializer:
        if t.data_type == BF16:
            a = numpy_helper.to_array(t).astype(np.float32)
            if a.size:
                wmax = max(wmax, float(np.abs(a).max()))
    print(f"  max |bf16 weight| = {wmax:g}  (fp16 max = 65504)")
    if wmax > 65504.0:
        print("\nERROR: some weights exceed fp16 range — refusing to convert (would inf).")
        sys.exit(1)

    print("Remapping bfloat16 → float16 …")
    counts = {"init": 0, "cast": 0, "attr": 0, "vinfo": 0}
    remap_graph_bf16_to_fp16(model.graph, counts)
    print(f"  converted: {counts['init']} initializers, "
          f"{counts['cast']} Cast targets, {counts['attr']} attr tensors, "
          f"{counts['vinfo']} value_info")

    # Output is > 2 GB → must use external data (protobuf single-file limit is 2 GB).
    print(f"Saving {out_path} (external data → {out_base}.data) …")
    stale = os.path.join(out_dir, out_base + ".data")
    if os.path.exists(stale):
        os.remove(stale)
    onnx.save(
        model,
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
    print("Compare against the bf16 original (watch the [FSB] timing backbone column):")
    print(f"  bf16:  ./build/fast_sam_3dbody_run --from input.mp4 --backbone {in_base}")
    print(f"  fp16:  ./build/fast_sam_3dbody_run --from input.mp4 --backbone {out_base}")


if __name__ == "__main__":
    main()
