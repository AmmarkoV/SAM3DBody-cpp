#!/usr/bin/env python3
"""build_decoder_heads.py — fuse norm_final + the MHR/cam regression heads into
one ONNX graph per refined-pose decoder variant.

Why
---
The --refined-pose loops run norm_final on the GPU, copy the 1024-dim pose token
back to the host, and then evaluate two 1024x1024->N MLPs on the CPU with a
scalar dot-product loop.  Measured on an RTX 4080 SUPER that CPU step costs
~9.2 ms per person per frame — more than the six transformer layers it sits
between — because each call streams ~10 MB of fp32 weights through one core.

The heads are plain Linear/ReLU/Linear, so they belong in the graph.  This script
appends them to the existing norm_final export, taking the weights from the same
GGUF files the C++ loads them from today.  Nothing is re-derived from the
original checkpoint: the output is bit-for-bit the same arithmetic, just run
where the token already is.

Produces (new names, alongside the existing decoder_*_normfinal.onnx which stay
valid as the fallback path):

  decoder_pass1_head.onnx     pass-1    norm_final + mhr_proj      + cam_proj
  decoder_prompted_head.onnx  pass-2    norm_final + mhr_proj      + cam_proj
  decoder_hand_head.onnx      hand crop norm_final + mhr_proj_hand + cam_proj_hand

Each takes the same `token` [B,N,1024] input as the norm_final graph it replaces
and produces three outputs — `pose_token` [B,1024], `mhr` [B,out], `cam` [B,3] —
so one graph serves both the intermediate iterations (which want mhr/cam) and the
final regress (which wants pose_token, and for pass 2 / hands mhr+cam as well).

Usage:
    tools/build_decoder_heads.py [--onnx-dir onnx] [--verify]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ── minimal GGUF v2/v3 reader ────────────────────────────────────────────────
# Only what is needed here: tensor shapes, dtypes and offsets.  Mirrors
# cffn_load()'s reader in src/core/fast_sam_3dbody.cpp (F32 and F16 only).

_GGML_F32, _GGML_F16 = 0, 1


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_kv_value(f, vtype):
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in fixed:
        f.read(fixed[vtype])
    elif vtype == 8:  # string
        _read_str(f)
    elif vtype == 9:  # array
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _skip_kv_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_tensors(path):
    """-> {name: numpy array}, shaped (ne[-1], ..., ne[0]) i.e. C order."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        alignment = 32
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vtype == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_kv_value(f, vtype)

        infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            infos.append((name, dims, ggml_type, offset))

        pos = f.tell()
        data_base = pos + (-pos) % alignment

        out = {}
        for name, dims, ggml_type, offset in infos:
            n = 1
            for d in dims:
                n *= d
            f.seek(data_base + offset)
            if ggml_type == _GGML_F32:
                arr = np.frombuffer(f.read(4 * n), dtype="<f4")
            elif ggml_type == _GGML_F16:
                arr = np.frombuffer(f.read(2 * n), dtype="<f2").astype(np.float32)
            else:
                raise ValueError(f"{name}: unsupported ggml type {ggml_type}")
            # GGUF dims are ne[0] first (fastest varying); numpy wants the reverse.
            out[name] = arr.reshape(tuple(reversed(dims))).copy()
        return out


# ── graph construction ───────────────────────────────────────────────────────

def load_ffn(tensors, prefix):
    """-> (W0, b0, W1, b1) with W shaped (out, in), matching linear_relu()."""
    try:
        w0 = tensors[f"{prefix}.fc0.weight"]
        b0 = tensors[f"{prefix}.fc0.bias"]
        w1 = tensors[f"{prefix}.fc1.weight"]
        b1 = tensors[f"{prefix}.fc1.bias"]
    except KeyError as e:
        raise SystemExit(f"missing FFN tensor {e} for prefix '{prefix}'")
    return w0, b0, w1, b1


def append_head(nodes, inits, src, name, ffn):
    """pose_token -> Gemm/Relu/Gemm -> `name`.  Gemm transB=1 consumes the
    (out, in) weight layout directly, so no transpose is materialised."""
    w0, b0, w1, b1 = ffn
    inits += [
        numpy_helper.from_array(w0.astype(np.float32), f"{name}.fc0.weight"),
        numpy_helper.from_array(b0.astype(np.float32), f"{name}.fc0.bias"),
        numpy_helper.from_array(w1.astype(np.float32), f"{name}.fc1.weight"),
        numpy_helper.from_array(b1.astype(np.float32), f"{name}.fc1.bias"),
    ]
    nodes += [
        helper.make_node("Gemm", [src, f"{name}.fc0.weight", f"{name}.fc0.bias"],
                         [f"{name}/fc0"], name=f"{name}/Gemm_0", transB=1),
        helper.make_node("Relu", [f"{name}/fc0"], [f"{name}/relu"], name=f"{name}/Relu"),
        helper.make_node("Gemm", [f"{name}/relu", f"{name}.fc1.weight", f"{name}.fc1.bias"],
                         [name], name=f"{name}/Gemm_1", transB=1),
    ]
    return w1.shape[0]


def build(normfinal_path, out_path, mhr_ffn, cam_ffn):
    src = onnx.load(normfinal_path)
    g = src.graph
    if len(g.output) != 1 or g.output[0].name != "pose_token":
        raise SystemExit(f"{normfinal_path}: expected a single 'pose_token' output")

    nodes = list(g.node)
    inits = list(g.initializer)
    mhr_dim = append_head(nodes, inits, "pose_token", "mhr", mhr_ffn)
    cam_dim = append_head(nodes, inits, "pose_token", "cam", cam_ffn)

    batch = g.output[0].type.tensor_type.shape.dim[0].dim_param or "B"
    outputs = [
        g.output[0],
        helper.make_tensor_value_info("mhr", TensorProto.FLOAT, [batch, mhr_dim]),
        helper.make_tensor_value_info("cam", TensorProto.FLOAT, [batch, cam_dim]),
    ]

    graph = helper.make_graph(nodes, g.name or "decoder_head",
                              list(g.input), outputs, inits)
    model = helper.make_model(graph, opset_imports=list(src.opset_import),
                              ir_version=src.ir_version)
    model.producer_name = "build_decoder_heads.py"
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return mhr_dim, cam_dim


# ── verification ─────────────────────────────────────────────────────────────

def verify(out_path, normfinal_path, mhr_ffn, cam_ffn):
    """Run the fused graph against a numpy replay of the C++ path
    (norm_final graph + linear_relu) on random tokens."""
    import onnxruntime as ort

    ref_sess = ort.InferenceSession(normfinal_path, providers=["CPUExecutionProvider"])
    # token count is baked into each variant's export (145 pass-1 / 148 pass-2 / …)
    n_tok = ref_sess.get_inputs()[0].shape[1]

    rng = np.random.default_rng(0)
    tok = rng.standard_normal((1, n_tok, 1024), dtype=np.float32)

    pose_ref = ref_sess.run(None, {"token": tok})[0]

    def mlp(x, ffn):
        w0, b0, w1, b1 = ffn
        h = np.maximum(0.0, x @ w0.T + b0)
        return h @ w1.T + b1

    mhr_ref, cam_ref = mlp(pose_ref, mhr_ffn), mlp(pose_ref, cam_ffn)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    pose, mhr, cam = sess.run(["pose_token", "mhr", "cam"], {"token": tok})

    return (float(np.abs(pose - pose_ref).max()),
            float(np.abs(mhr - mhr_ref).max()),
            float(np.abs(cam - cam_ref).max()))


VARIANTS = [
    # (norm_final source,              output,                       gguf,                    mhr prefix,      cam prefix)
    ("decoder_pass1_normfinal.onnx",    "decoder_pass1_head.onnx",    "pipeline.gguf",         "mhr_proj",      "cam_proj"),
    ("decoder_prompted_normfinal.onnx", "decoder_prompted_head.onnx", "pipeline.gguf",         "mhr_proj",      "cam_proj"),
    ("decoder_hand_normfinal.onnx",     "decoder_hand_head.onnx",     "pipeline_refined.gguf", "mhr_proj_hand", "cam_proj_hand"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", default=str(Path(__file__).resolve().parent.parent / "onnx"))
    ap.add_argument("--verify", action="store_true",
                    help="check each fused graph against a numpy replay of the CPU path")
    args = ap.parse_args()

    d = Path(args.onnx_dir)
    cache = {}
    rc = 0
    for nf, out, gguf, mhr_p, cam_p in VARIANTS:
        nf_path, out_path, gguf_path = d / nf, d / out, d / gguf
        for p in (nf_path, gguf_path):
            if not p.exists():
                print(f"skip {out}: missing {p.name}", file=sys.stderr)
                break
        else:
            if gguf_path not in cache:
                cache[gguf_path] = gguf_tensors(gguf_path)
            t = cache[gguf_path]
            mhr_ffn, cam_ffn = load_ffn(t, mhr_p), load_ffn(t, cam_p)
            mhr_dim, cam_dim = build(str(nf_path), str(out_path), mhr_ffn, cam_ffn)
            size = out_path.stat().st_size
            msg = f"{out}: mhr[{mhr_dim}] cam[{cam_dim}]  {size/1e6:.1f} MB"
            if args.verify:
                dp, dm, dc = verify(str(out_path), str(nf_path), mhr_ffn, cam_ffn)
                msg += f"  max|diff| pose={dp:.3e} mhr={dm:.3e} cam={dc:.3e}"
                if max(dp, dm, dc) > 1e-3:
                    msg += "  ** MISMATCH **"
                    rc = 1
            print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
