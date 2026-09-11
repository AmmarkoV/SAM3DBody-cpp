#!/usr/bin/env python3
"""Sanity-check / smoke-test a .arfz container written by ARFWriter (arf_writer.cpp).

Matches the conformant-rewrite shape adopted 2026-09-11 (numeric component
ids, structure.assets[].lods[], data[].type, big-endian AAU stream,
BlendshapeSet.shapes as GLB) — see knowledge/ARF.md. For a real independent
check, prefer AmmarkoV/ARFPlayer's own tools (arfinfo_cpp, `arfplay --info`),
which read the exact same shape this writer produces and have been checked
against the ISO/IEC 23090-39 FDIS-stage text; this script is a lighter,
dependency-free smoke test that lives in this repo.

It unzips the container, checks the mandatory arf.json top-level keys,
cross-checks the skeleton's joint count/hierarchy, decodes the AAU_JOINT
animation stream, and reports the root joint's per-frame translation range —
compare that against the corresponding --bvh run's printed "root path
X=[...] Y=[...] Z=[...]" line as an independent cross-check (both should
match to within float noise; see knowledge/ARF.md "Root translation is the
camera-translation head").

This reads exactly the binary layout arf_writer.cpp writes (documented in
knowledge/ARF.md) — it is not a general ARF/ISO-23090-39 parser, and it does
not decode the face blendshapes' GLB geometry (only counts AAU_BLENDSHAPE
frames).

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
    """Big-endian AAU framing: (unit_type<<1)|reserved byte, uint32BE length."""
    off = 0
    while off < len(data):
        unit_type = data[off] >> 1
        off += 1
        (unit_len,) = struct.unpack_from(">I", data, off)
        off += 4
        payload = data[off:off + unit_len]
        off += unit_len
        yield AAU(unit_type, payload)


def parse_config_payload(payload: bytes):
    (_ts,) = struct.unpack_from(">I", payload, 0)
    plen = payload[4]
    profile = payload[5:5 + plen].decode("utf-8")
    (timescale,) = struct.unpack_from(">f", payload, 5 + plen)
    return profile, timescale


def parse_joint_payload(payload: bytes):
    (ts,) = struct.unpack_from(">I", payload, 0)
    (joint_set_id,) = struct.unpack_from(">H", payload, 4)
    flags = payload[6]
    (count_minus1,) = struct.unpack_from(">H", payload, 7)
    n_joints = count_minus1 + 1
    velocity_present = bool(flags & 0x80)
    p = 9
    mats = {}
    for _ in range(n_joints):
        (jidx,) = struct.unpack_from(">H", payload, p)
        p += 2
        mat = struct.unpack_from(">16f", payload, p)
        p += 64
        if velocity_present:
            p += 64
        mats[jidx] = mat
    return ts, joint_set_id, mats


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

        assets = doc["structure"].get("assets", [])
        if not assets:
            fail("structure.assets is empty")
        lod = assets[0]["lods"][0]
        print(f"OK  structure: asset={assets[0].get('name')!r} lod={lod.get('name')!r} {lod}")

        comp = doc["components"]
        nodes = comp.get("nodes", [])
        skeleton = comp.get("skeletons", [{}])[0]
        skel_joint_ids = skeleton.get("joints", [])
        print(f"OK  skeleton: {len(nodes)} nodes, {len(skel_joint_ids)} skeleton joints, "
              f"root={skeleton.get('root')!r}")
        if len(nodes) != len(skel_joint_ids):
            fail(f"node count ({len(nodes)}) != skeleton joint count ({len(skel_joint_ids)})")

        # Every reference resolves by matching declared numeric `id`, never by
        # array position — see knowledge/ARF.md / the spec's General
        # Conventions clause.
        nodes_by_id = {n["id"]: n for n in nodes}
        if skel_joint_ids != sorted(skel_joint_ids):
            fail("skeletons[0].joints is not sorted — this writer always assigns id==index")
        for n in nodes:
            parent = n.get("parent")
            if parent is not None and parent not in nodes_by_id:
                fail(f"node {n['id']!r} has unresolved parent {parent!r}")
        if skeleton.get("root") not in nodes_by_id:
            fail(f"skeletons[0].root {skeleton.get('root')!r} does not resolve to a node id")
        print("OK  all node parent / skeleton root references resolve")

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

        mesh = comp.get("meshes", [{}])[0]
        positions_id, indices_id = mesh.get("data", [None, None])
        if positions_id in data_items:
            dims, dtype, blob = parse_dense_tensor(zf.read(data_items[positions_id]["uri"]))
            n = dims[0] * dims[1]
            verts = struct.unpack_from(f"<{n}f", blob, 0)
            xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
            print(f"OK  mesh positions (data id {positions_id}): dims={dims} dtype={dtype} "
                  f"bbox X=[{min(xs):.2f},{max(xs):.2f}] Y=[{min(ys):.2f},{max(ys):.2f}] "
                  f"Z=[{min(zs):.2f},{max(zs):.2f}]")

        skin = comp.get("skins", [{}])[0]
        weights_id = skin.get("weights")
        if weights_id in data_items:
            dims, value_count, idx, vals = parse_sparse_tensor(zf.read(data_items[weights_id]["uri"]))
            n_joints = dims[1]
            sums: dict[int, float] = {}
            for flat, w in zip(idx, vals):
                v = flat // n_joints
                sums[v] = sums.get(v, 0.0) + w
            bad = [v for v, s in sums.items() if not (0.99 <= s <= 1.01)]
            print(f"OK  skin_weights (data id {weights_id}): dims={dims} valueCount={value_count} "
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
        if joint_frames and joint_frames[0][1] != skeleton.get("id"):
            fail(f"AAU_JOINT joint_set_id {joint_frames[0][1]!r} != "
                 f"skeletons[0].id {skeleton.get('id')!r}")

        if joint_frames:
            root_idx = skeleton.get("root")
            txs = [f[2][root_idx][3] for f in joint_frames if root_idx in f[2]]
            tys = [f[2][root_idx][7] for f in joint_frames if root_idx in f[2]]
            tzs = [f[2][root_idx][11] for f in joint_frames if root_idx in f[2]]
            if txs:
                print(f"OK  root (node id {root_idx}) translation range: "
                      f"X=[{min(txs):.1f},{max(txs):.1f}] "
                      f"Y=[{min(tys):.1f},{max(tys):.1f}] "
                      f"Z=[{min(tzs):.1f},{max(tzs):.1f}]  "
                      f"(compare against the matching --bvh run's printed root path)")

            # Placement in the declared frame: Y-up, -Z forward, origin at the
            # camera (see ARF.md "Coordinate convention").  Composing the first
            # frame's joint origins catches a sign error in the root translation,
            # which is invisible in the pose alone — the skeleton stays perfectly
            # upright while the whole body sits behind and above the camera.
            parent_of = {n["id"]: n.get("parent") for n in nodes}
            mats0 = joint_frames[0][2]
            world = {}

            def compose(nid):
                if nid in world:
                    return world[nid]
                m = mats0.get(nid)
                if m is None:
                    return None
                par = parent_of.get(nid)
                if par is None:
                    world[nid] = m
                else:
                    pm = compose(par)
                    if pm is None:
                        return None
                    world[nid] = tuple(
                        sum(pm[4 * i + k] * m[4 * k + j] for k in range(4))
                        for i in range(4) for j in range(4)
                    )
                return world[nid]

            pts = [(w[3], w[7], w[11]) for w in
                   (compose(nid) for nid in mats0) if w is not None]
            if pts:
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                if max(zs) >= 0.0:
                    fail(f"frame 0 has joints at Z >= 0 (Z range [{min(zs):.1f},{max(zs):.1f}]) — "
                         "the subject must lie in front of a camera that looks down -Z. "
                         "A positive Z means the root translation was not negated "
                         "(see ARF.md 'Coordinate convention').")
                if min(ys) >= 0.0:
                    fail(f"frame 0 is entirely above the origin (Y range [{min(ys):.1f},{max(ys):.1f}]) — "
                         "the subject's feet should sit below a camera at standing height. "
                         "This is the signature of an un-negated root translation Y.")
                straddles = min(ys) < 0.0 < max(ys)
                print(f"OK  frame 0 placement: Y=[{min(ys):.1f},{max(ys):.1f}] "
                      f"Z=[{min(zs):.1f},{max(zs):.1f}] cm"
                      + ("" if straddles else
                         "  WARNING: body does not straddle Y=0 — plausible only if the "
                         "camera was below the feet or above the head"))

        blendshape_sets = comp.get("blendshapeSets", [])
        if blendshape_sets:
            bs = blendshape_sets[0]
            shape_ids = bs.get("shapes", [])
            missing_shapes = [sid for sid in shape_ids if sid not in data_items]
            if missing_shapes:
                fail(f"blendshapeSets[0].shapes references missing data id(s): {missing_shapes}")
            print(f"OK  blendshapeSet {bs.get('name')!r}: {len(shape_ids)} shape GLB(s) present")

        if "animations/face.bin" in names:
            face_stream = list(read_aau_stream(zf.read("animations/face.bin")))
            n_bs = sum(1 for a in face_stream if a.unit_type == AAU_BLENDSHAPE)
            print(f"OK  face.bin present: {n_bs} AAU_BLENDSHAPE frame(s)")

        if "id_map.txt" in names:
            n_lines = zf.read("id_map.txt").count(b"\n")
            print(f"OK  id_map.txt present: {n_lines} line(s)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
