#!/usr/bin/env python3
"""
tools/export_backbone_fp32.py  —  re-export backbone.onnx in pure float32

WHY THIS EXISTS
---------------
The standard backbone.onnx (produced by fast_sam_3dbody_cpp/export_onnx.py
with --bf16) contains 224 Cast-to-bfloat16 nodes.  The Expand ops that
broadcast cls_token / storage_tokens / rope embeddings then operate on
bfloat16 tensors, and ONNX Runtime's CPU execution provider has no kernel
for bfloat16 Expand:

  ORT error: Could not find an implementation for Expand(13) node with name '/Expand'

This script re-exports the backbone in float32 only, producing
backbone_fp32.onnx that loads and runs on ORT CPU EP.

PERFORMANCE NOTE
----------------
DINOv3-ViT-H has ~630 M parameters.  On a modern laptop CPU, one forward
pass takes roughly 5–15 seconds vs ~150 ms on an RTX 3090.  This model is
not designed for CPU inference; backbone_fp32.onnx is a compatibility shim
for machines without CUDA, not a speed solution.

USAGE
-----
  # Run from the Fast-SAM-3D-Body project root with the training venv active:
  python /path/to/SAM3DBody-cpp/tools/export_backbone_fp32.py \\
      --checkpoint ./checkpoints/sam-3d-body-dinov3 \\
      --output     /path/to/SAM3DBody-cpp/onnx

  # Then run the C++ binary with the new backbone:
  ./build/fast_sam_3dbody_run --backbone backbone_fp32.onnx --from video.mp4

  # On a GPU machine, still prefer backbone.onnx (BF16, 3x faster):
  ./build/fast_sam_3dbody_run --backbone backbone.onnx --from video.mp4

DEPENDENCIES
------------
  torch torchvision onnx onnxruntime
  The full Fast-SAM-3D-Body Python environment (sam_3d_body package, etc.)
  — this must be run from the Fast-SAM-3D-Body repo root.
"""

import argparse
import contextlib
import os
import sys
import types

# Environment flags must be set before the model is imported.
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")

import torch
import torch.nn as nn

IMAGE_SIZE   = 512
BACKBONE_DIM = 1280


class BackboneWrapper(nn.Module):
    """DINOv3 backbone: [B,3,512,512] → [B,1280,32,32], no bfloat16."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_intermediate_layers(
            x, n=1, reshape=True, norm=True
        )[-1]


def _patch_encoder_fp32(encoder: nn.Module) -> None:
    """Minimal patch: remove mask_token from prepare_tokens_with_masks.

    This is the same graph-cleanup patch as export_onnx.py uses, but
    WITHOUT the _Float32Conv wrapper (patch_embed stays float32) and
    WITHOUT _patch_swiglu_silu_float32 (not needed for CPU export).
    """
    def _prepare_tokens_patched(self, x, masks=None):
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = self.cls_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1],
                dtype=cls_token.dtype, device=cls_token.device,
            )
        x = torch.cat([
            cls_token.expand(B, -1, -1),
            storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)
        return x, (H, W)

    encoder.prepare_tokens_with_masks = types.MethodType(
        _prepare_tokens_patched, encoder
    )


def export_backbone_fp32(model, out_dir: str, opset: int = 18,
                         device: str = "cpu"):
    path = os.path.join(out_dir, "backbone_fp32.onnx")
    print(f"\n── backbone_fp32 → {path}  (float32, {device})")

    encoder = model.backbone.encoder
    encoder.float()   # weights may load as bfloat16; cast all params to float32
    _patch_encoder_fp32(encoder)
    wrapper = BackboneWrapper(encoder)
    wrapper.eval().to(device)

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE,
                        device=device, dtype=torch.float32)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"   in {tuple(dummy.shape)}  out {tuple(out.shape)}  dtype={out.dtype}")

    if out.dtype != torch.float32:
        raise RuntimeError(
            f"Output is {out.dtype}, not float32 — check that BF16 autocast "
            "is fully disabled."
        )

    # Export to a temp path first, then consolidate external data.
    # torch.onnx.export scatters one file per tensor by default; we merge them
    # into a single backbone_fp32.onnx.data file to match backbone.onnx layout.
    import tempfile, shutil, onnx
    from onnx.external_data_helper import convert_model_to_external_data

    tmp_dir = tempfile.mkdtemp(prefix="fsb_fp32_export_")
    tmp_onnx = os.path.join(tmp_dir, "backbone_fp32.onnx")
    try:
        torch.onnx.export(
            wrapper, dummy, tmp_onnx,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={"image": {0: "B"}, "features": {0: "B"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

        print("   Consolidating external data into single .data file …")
        model = onnx.load(tmp_onnx)
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location="backbone_fp32.onnx.data",
            size_threshold=0,
            convert_attribute=False,
        )
        onnx.save_model(model, path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = (os.path.getsize(path) +
               os.path.getsize(path + ".data")) / 1e6
    print(f"   {size_mb:.0f} MB  ✓  ({path}, {path}.data)")

    # Verify no bfloat16 nodes survived
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        bf16_casts = sum(
            1 for n in m.graph.node
            if n.op_type == "Cast"
            and any(a.name == "to" and a.i == 16 for a in n.attribute)
        )
        if bf16_casts:
            print(f"   WARNING: {bf16_casts} Cast-to-bfloat16 nodes still present"
                  " — CPU EP may still fail")
        else:
            print("   Verified: no bfloat16 Cast nodes — safe for ORT CPU EP")
    except ImportError:
        pass  # onnx not installed; skip check

    # Quick ORT CPU EP smoke-test
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        import numpy as np
        inp = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        out_ort = sess.run(None, {"image": inp})[0]
        print(f"   ORT CPU EP smoke-test: output {out_ort.shape}  ✓")
    except Exception as e:
        print(f"   ORT CPU EP smoke-test FAILED: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Export backbone in float32 for ORT CPU EP compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", default="./checkpoints/sam-3d-body-dinov3",
                    help="Checkpoint directory (contains model.ckpt)")
    ap.add_argument("--output", default="./fast_sam_3dbody_cpp/onnx",
                    help="Output directory (default: ./fast_sam_3dbody_cpp/onnx)")
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset version (default: 18)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for tracing (default: cpu; use cuda for faster export)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model …  (this takes a minute)")
    # sys.path must include the Fast-SAM-3D-Body root so sam_3d_body is importable.
    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from sam_3d_body.build_models import load_sam_3d_body
    ckpt = os.path.join(args.checkpoint, "model.ckpt")
    mhr  = os.path.join(args.checkpoint, "assets", "mhr_model.pt")

    if not os.path.exists(ckpt):
        print(f"ERROR: {ckpt} not found")
        sys.exit(1)

    model, _ = load_sam_3d_body(checkpoint_path=ckpt, mhr_path=mhr)
    model.eval()
    print("Model loaded.")

    export_backbone_fp32(model, args.output, opset=args.opset, device=args.device)

    print("\n✓ Done.")
    print(f"  backbone_fp32.onnx is in: {os.path.abspath(args.output)}")
    print()
    print("To use with the C++ binary:")
    print("  ./build/fast_sam_3dbody_run --backbone backbone_fp32.onnx --from video.mp4 --cuda -1")
    print()
    print("Expected CPU throughput: ~5–15 s/frame on a typical laptop (ViT-H is large).")
    print("For real-time use, a CUDA GPU and the BF16 backbone.onnx are required.")


if __name__ == "__main__":
    main()
