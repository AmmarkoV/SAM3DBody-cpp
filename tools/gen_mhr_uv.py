#!/usr/bin/env python3
"""Generate a UV atlas for the MHR body mesh (onnx/body_mesh.tri) -> onnx/body_mesh_uv.bin.

Offline tool (needs `pip install xatlas numpy`); the C++ runtime only reads the output.
The atlas splits vertices along UV seams, so the output describes "wedges" (a vertex
with one UV) that point back at the original vertex they came from; skinning keeps
running on the original 18439 vertices and the wedges just look their positions up.

Output layout (little endian):
    char     magic[8]   "MHRUV001"
    uint32   n_wedges, n_tris, atlas_w, atlas_h
    float32  uv[n_wedges * 2]        in [0,1], v = 0 at the top row of the texture
    uint32   vref[n_wedges]          original vertex of each wedge
    uint32   indices[n_tris * 3]     triangles over wedges, same order as body_mesh.tri

Usage: tools/gen_mhr_uv.py [onnx/body_mesh.tri] [onnx/body_mesh_uv.bin] [--resolution 1024]
"""
import argparse
import struct

import numpy as np
import xatlas

TRI_HEADER = 136   # sizeof(struct TRI_Header): char[5] + 3 padding, 11 uint32, float[16], 5 uint32


def load_tri(path):
    data = open(path, "rb").read()
    assert data[:5] == b"TRI3D", "not a TRI file"
    (tri_type, name_size, float_size, draw_type, n_vert, n_norm, n_tex, n_col, n_idx,
     n_bones, root) = struct.unpack_from("<11I", data, 8)
    off = TRI_HEADER + name_size
    verts = np.frombuffer(data, np.float32, n_vert, off).reshape(-1, 3)
    off += 4 * (n_vert + n_norm + n_tex + n_col)
    idx = np.frombuffer(data, np.uint32, n_idx, off).reshape(-1, 3)
    return verts.copy(), idx.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tri", nargs="?", default="onnx/body_mesh.tri")
    ap.add_argument("out", nargs="?", default="onnx/body_mesh_uv.bin")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--padding", type=int, default=4, help="texels between charts")
    a = ap.parse_args()

    verts, faces = load_tri(a.tri)
    print(f"mesh: {len(verts)} vertices, {len(faces)} triangles")

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    co = xatlas.ChartOptions()
    po = xatlas.PackOptions()
    po.resolution = a.resolution
    po.padding = a.padding
    po.bilinear = True
    po.rotate_charts = True
    atlas.generate(co, po)
    vmap, ind, uv = atlas[0]
    print(f"atlas: {atlas.width}x{atlas.height}, {atlas.chart_count} charts, "
          f"{len(vmap)} wedges ({len(vmap) - len(verts)} added at seams), "
          f"utilization {float(np.atleast_1d(atlas.utilization)[0]):.2f}")
    assert len(ind) == len(faces)
    uv = np.asarray(uv, np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]                 # v = 0 at the top row, like an image
    with open(a.out, "wb") as f:
        f.write(b"MHRUV001")
        f.write(struct.pack("<4I", len(vmap), len(ind), atlas.width, atlas.height))
        f.write(uv.astype("<f4").tobytes())
        f.write(np.asarray(vmap, "<u4").tobytes())
        f.write(np.asarray(ind, "<u4").tobytes())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
