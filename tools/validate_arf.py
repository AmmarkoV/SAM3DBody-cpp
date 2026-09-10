#!/usr/bin/env python3
"""Sanity-check / smoke-test a .arfz container written by ARFWriter (arf_writer.cpp).

There is no ARF viewer available to this project, so this script is the
practical way to catch a malformed writer: it unzips the container, checks
the mandatory arf.json top-level keys, cross-checks the skeleton's joint
count/hierarchy, decodes the AAU_JOINT animation stream, and reports the
root joint's per-frame translation range — compare that against the
corresponding --bvh run's printed "root path X=[...] Y=[...] Z=[...]" line
as an independent cross-check (both should match to within float noise; see
knowledge/ARF.md "Root translation is the camera-translation head").

This reads exactly the binary layout arf_writer.cpp writes (documented in
knowledge/ARF.md) — it is not a general ARF/ISO-23090-39 parser.

Usage:
    python3 tools/validate_arf.py path/to/person_0.arfz [--verbose]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from dataclasses import dataclass


REQUIRED_TOP_LEVEL_KEYS = ("preamble", "metadata", "structure", "components", "data")

AAU_CONFIG = 0
AAU_BLENDSHAPE = 1
AAU_JOINT = 2


@dataclass
class AAU:
    unit_type: int
    payload: bytes


def read_aau_stream(data: bytes):
    off = 0
    while off < len(data):
        unit_type = data[off]
        off += 1
        (unit_len,) = struct.unpack_from("<I", data, off)
        off += 4
        payload = data[off:off + unit_len]
        off += unit_len
        yield AAU(unit_type, payload)


def parse_config_payload(payload: bytes):
    (_ts,) = struct.unpack_from("<I", payload, 0)
    (plen,) = struct.unpack_from("<I", payload, 4)
    profile = payload[8:8 + plen].decode("utf-8")
    (timescale,) = struct.unpack_from("<f", payload, 8 + plen)
    return profile, timescale


def parse_joint_payload(payload: bytes):
    (ts,) = struct.unpack_from("<I", payload, 0)
    (n_joints,) = struct.unpack_from("<I", payload, 4)
    p = 8
    mats = {}
    for _ in range(n_joints):
        (jidx,) = struct.unpack_from("<I", payload, p)
        p += 4
        mat = struct.unpack_from("<16f", payload, p)
        p += 64
        mats[jidx] = mat
    return ts, mats


def parse_dense_tensor(data: bytes):
    (ndims,) = struct.unpack_from("<i", data, 0)
    dims = struct.unpack_from(f"<{ndims}i", data, 4)
    off = 4 + 4 * ndims
    (dtype,) = struct.unpack_from("<i", data, off)
    off += 4
    return dims, dtype, data[off:]


def parse_sparse_tensor(data: bytes):
    (ndims,) = struct.unpack_from("<i", data, 0)
    dims = struct.unpack_from(f"<{ndims}i", data, 4)
    off = 4 + 4 * ndims
    (value_count,) = struct.unpack_from("<i", data, off); off += 4
    (itype,) = struct.unpack_from("<i", data, off); off += 4
    (dtype,) = struct.unpack_from("<i", data, off); off += 4
    idx = struct.unpack_from(f"<{value_count}I", data, off); off += 4 * value_count
    vals = struct.unpack_from(f"<{value_count}f", data, off)
    return dims, value_count, idx, vals


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arfz", help="path to a .arfz container")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    with zipfile.ZipFile(args.arfz) as zf:
        names = set(zf.namelist())
        if "arf.json" not in names:
            fail("arf.json missing from archive root")

        doc = json.loads(zf.read("arf.json"))
        missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in doc]
        if missing:
            fail(f"arf.json missing required top-level keys: {missing}")
        print(f"OK  arf.json: {list(doc.keys())}")
        print(f"    preamble: {doc['preamble']}")

        comp = doc["components"]
        nodes = comp.get("nodes", [])
        skeleton = comp.get("skeletons", [{}])[0]
        n_skel_joints = len(skeleton.get("joints", []))
        print(f"OK  skeleton: {len(nodes)} nodes, {n_skel_joints} skeleton joints, "
              f"root={skeleton.get('root')!r}")
        if len(nodes) != n_skel_joints:
            fail(f"node count ({len(nodes)}) != skeleton joint count ({n_skel_joints})")

        # Parent references must resolve to another node id (or be absent = root).
        node_ids = {n["id"] for n in nodes}
        for n in nodes:
            parent = n.get("parent")
            if parent is not None and parent not in node_ids:
                fail(f"node {n['id']!r} has unresolved parent {parent!r}")
        print("OK  all node parent references resolve")

        data_items = {d["id"]: d for d in doc.get("data", [])}
        for item_id, item in data_items.items():
            uri = item["uri"]
            if uri not in names:
                fail(f"data item {item_id!r} references missing file {uri!r}")
            actual = len(zf.read(uri))
            if actual != item.get("byteLength"):
                fail(f"data item {item_id!r}: byteLength {item.get('byteLength')} "
                     f"!= actual file size {actual}")
        print(f"OK  {len(data_items)} data item(s), all present with matching byteLength")

        if "mesh_positions" in data_items:
            dims, dtype, blob = parse_dense_tensor(zf.read(data_items["mesh_positions"]["uri"]))
            n = dims[0] * dims[1]
            verts = struct.unpack_from(f"<{n}f", blob, 0)
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            print(f"OK  mesh_positions: dims={dims} dtype={dtype} "
                  f"bbox X=[{min(xs):.2f},{max(xs):.2f}] Y=[{min(ys):.2f},{max(ys):.2f}] "
                  f"Z=[{min(zs):.2f},{max(zs):.2f}]")

        if "skin_weights" in data_items:
            dims, value_count, idx, vals = parse_sparse_tensor(zf.read(data_items["skin_weights"]["uri"]))
            n_joints = dims[1]
            sums: dict[int, float] = {}
            for flat, w in zip(idx, vals):
                v = flat // n_joints
                sums[v] = sums.get(v, 0.0) + w
            bad = [v for v, s in sums.items() if not (0.99 <= s <= 1.01)]
            print(f"OK  skin_weights: dims={dims} valueCount={value_count} "
                  f"{len(sums)} verts weighted, {len(bad)} with sum outside [0.99,1.01]")
            if bad and args.verbose:
                print(f"    bad vertex ids (first 10): {bad[:10]}")

        if "animations/joints.bin" not in names:
            fail("animations/joints.bin missing")
        stream = list(read_aau_stream(zf.read("animations/joints.bin")))
        if not stream or stream[0].unit_type != AAU_CONFIG:
            fail("joints.bin does not start with an AAU_CONFIG unit")
        profile, timescale = parse_config_payload(stream[0].payload)
        print(f"OK  joints.bin config: profile={profile!r} timescale={timescale:.3f} ticks/s")

        joint_frames = [parse_joint_payload(a.payload) for a in stream[1:] if a.unit_type == AAU_JOINT]
        print(f"OK  {len(joint_frames)} AAU_JOINT frame(s), "
              f"~{len(joint_frames)/timescale:.2f} s duration")

        if joint_frames:
            root_id = skeleton.get("root")
            root_idx = next((i for i, n in enumerate(nodes) if n["id"] == root_id), None)
            if root_idx is not None:
                txs = [f[1][root_idx][3] for f in joint_frames if root_idx in f[1]]
                tys = [f[1][root_idx][7] for f in joint_frames if root_idx in f[1]]
                tzs = [f[1][root_idx][11] for f in joint_frames if root_idx in f[1]]
                if txs:
                    print(f"OK  root ({root_id!r}) translation range: "
                          f"X=[{min(txs):.1f},{max(txs):.1f}] "
                          f"Y=[{min(tys):.1f},{max(tys):.1f}] "
                          f"Z=[{min(tzs):.1f},{max(tzs):.1f}]  "
                          f"(compare against the matching --bvh run's printed root path)")

        if "animations/face.bin" in names:
            face_stream = list(read_aau_stream(zf.read("animations/face.bin")))
            n_bs = sum(1 for a in face_stream if a.unit_type == AAU_BLENDSHAPE)
            print(f"OK  face.bin present: {n_bs} AAU_BLENDSHAPE frame(s)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
