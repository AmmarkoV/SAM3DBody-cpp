#!/usr/bin/env python3
"""Inspect the BVH ↔ MHR rest-pose alignment dumped by bvh_writer.

Reads /tmp/bvh_writer_skeletons.csv (run the render binary with
BVH_WRITER_DUMP=1 to produce it) and reports:

  • Bounding boxes of each skeleton (so we can see scale/axis mismatches).
  • Procrustes-aligned scale/rotation/translation that maps MHR onto BVH.
  • Per-BVH-joint nearest MHR partner under the *aligned* MHR positions.

Usage:
    python3 tools/analyze_skeletons.py
"""
from __future__ import annotations
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

CSV_PATH = Path("/tmp/bvh_writer_skeletons.csv")


@dataclass
class Joint:
    name: str
    parent: int
    pos: np.ndarray  # (3,)


def load(csv_path: Path) -> Tuple[List[Joint], List[Joint]]:
    bvh: List[Joint] = []
    mhr: List[Joint] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            j = Joint(
                name=row["name"],
                parent=int(row["parent"]),
                pos=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
            )
            (bvh if row["side"] == "bvh" else mhr).append(j)
    return bvh, mhr


def bbox(points: np.ndarray) -> str:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return f"X[{lo[0]:7.2f},{hi[0]:7.2f}]  Y[{lo[1]:7.2f},{hi[1]:7.2f}]  Z[{lo[2]:7.2f},{hi[2]:7.2f}]"


def procrustes(A: np.ndarray, B: np.ndarray):
    """Best-fit similarity transform mapping B → A: A ≈ s · R @ B + t.
    Returns (s, R, t, aligned_B, rmse).
    """
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    norm_b = np.linalg.norm(B0)
    H = B0.T @ A0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T)
    s = np.trace(D @ np.diag(np.linalg.svd(H, compute_uv=False))) / (norm_b ** 2)
    aligned = s * (B0 @ R.T) + mu_a
    rmse = float(np.sqrt(((aligned - A) ** 2).sum(1).mean()))
    t = mu_a - s * (R @ mu_b)
    return s, R, t, aligned, rmse


def report_aligned_matches(bvh: List[Joint], mhr_aligned: np.ndarray, mhr: List[Joint], k: int = 20):
    """For each BVH joint, find the closest MHR joint after alignment."""
    tree = cKDTree(mhr_aligned)
    rows = []
    for j in bvh:
        d, idx = tree.query(j.pos, k=1)
        rows.append((j.name, idx, d))
    rows.sort(key=lambda r: r[2])
    print(f"\n  {'BVH joint':<28} {'MHR':>4}   d (cm)")
    for name, idx, d in rows[:k]:
        print(f"  {name:<28} {idx:>4}   {d:8.3f}")
    print("  ...")
    for name, idx, d in rows[-5:]:
        print(f"  {name:<28} {idx:>4}   {d:8.3f}")


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing {CSV_PATH}; run with BVH_WRITER_DUMP=1", file=sys.stderr)
        return 1

    bvh, mhr = load(CSV_PATH)
    bvh_pts = np.array([j.pos for j in bvh])
    mhr_pts = np.array([j.pos for j in mhr])

    print(f"BVH joints (named, channels): {len(bvh)}")
    print(f"MHR joints                  : {len(mhr)}")
    print(f"BVH bbox: {bbox(bvh_pts)}")
    print(f"MHR bbox: {bbox(mhr_pts)}")

    # ----- naive nearest neighbour BEFORE any alignment ---------------------
    tree = cKDTree(mhr_pts)
    raw_d = [tree.query(b.pos, k=1)[0] for b in bvh]
    raw_d = np.array(raw_d)
    print(f"\nRaw NN distance (no align): median={np.median(raw_d):.2f} cm   "
          f"p90={np.percentile(raw_d, 90):.2f}   max={raw_d.max():.2f}")

    # ----- pick a small anchor set of joints that we expect to exist in both
    # skeletons under any reasonable convention and procrustes-align using them
    anchor_names = [
        "hip", "abdomen", "chest", "neck", "head",
        "lShldr", "rShldr", "lForeArm", "rForeArm", "lHand", "rHand",
        "lThigh", "rThigh", "lShin", "rShin", "lFoot", "rFoot",
        "lCollar", "rCollar", "lButtock", "rButtock",
    ]
    by_name = {j.name: j for j in bvh}
    anchors = [(name, by_name[name].pos) for name in anchor_names if name in by_name]
    if len(anchors) < 6:
        print("not enough anchor joints in BVH side, exiting")
        return 1
    anchor_bvh = np.array([p for _, p in anchors])

    # We don't know which MHR joints are which — but we can try all sufficiently
    # large subsets. Simpler shortcut: assume hip→mhr0 (root) and align the
    # *whole* skeletons using a brute-force best-anchor mapping based on nearest
    # neighbor under successive scale guesses.
    print("\n--- Procrustes alignment using best NN-mapped anchors ---")
    # Just try a couple of common conventions:
    for label, transform in [
        ("identity", lambda p: p),
        ("flip Y",   lambda p: p * np.array([1, -1,  1])),
        ("flip Z",   lambda p: p * np.array([1,  1, -1])),
        ("flip Y+Z", lambda p: p * np.array([1, -1, -1])),
        ("flip X+Z", lambda p: p * np.array([-1, 1, -1])),
        ("flip X+Y", lambda p: p * np.array([-1, -1, 1])),
        ("flip X",   lambda p: p * np.array([-1, 1, 1])),
    ]:
        mhr_t = transform(mhr_pts)
        tree2 = cKDTree(mhr_t)
        d_med = float(np.median([tree2.query(b.pos, k=1)[0] for b in bvh]))
        print(f"  {label:<10}  bbox: {bbox(mhr_t):<60}  median NN d = {d_med:.2f} cm")

    # Use the BEST of those and then run procrustes for fine alignment
    best_label, best_transform, best_med = "identity", lambda p: p, np.inf
    for label, tr in [
        ("identity", lambda p: p),
        ("flip Y",   lambda p: p * np.array([1, -1,  1])),
        ("flip Z",   lambda p: p * np.array([1,  1, -1])),
        ("flip Y+Z", lambda p: p * np.array([1, -1, -1])),
        ("flip X+Z", lambda p: p * np.array([-1, 1, -1])),
        ("flip X+Y", lambda p: p * np.array([-1, -1, 1])),
        ("flip X",   lambda p: p * np.array([-1, 1, 1])),
    ]:
        mhr_t = tr(mhr_pts)
        tree2 = cKDTree(mhr_t)
        d_med = float(np.median([tree2.query(b.pos, k=1)[0] for b in bvh]))
        if d_med < best_med:
            best_label, best_transform, best_med = label, tr, d_med
    print(f"\nBest naive axis flip: {best_label!r}  (median NN d = {best_med:.2f} cm)")

    mhr_t = best_transform(mhr_pts)

    # Build paired anchors using nearest-neighbor under the chosen flip.
    tree2 = cKDTree(mhr_t)
    bvh_anchor_pts = []
    mhr_anchor_pts = []
    for name, p in anchors:
        d, idx = tree2.query(p, k=1)
        if d < 30.0:  # generous: anchors expected to align coarsely
            bvh_anchor_pts.append(p)
            mhr_anchor_pts.append(mhr_t[idx])
    if len(bvh_anchor_pts) < 6:
        print("anchor pairing too sparse; aborting procrustes")
        return 1

    A = np.array(bvh_anchor_pts)
    B = np.array(mhr_anchor_pts)
    s, R, t, _, rmse = procrustes(A, B)
    print(f"\nProcrustes (after {best_label}): s={s:.4f}  rmse={rmse:.3f} cm")
    print(f"R =\n{R}")
    print(f"t = {t}")

    # Apply the full transform to all MHR joints and report matches.
    mhr_aligned = s * (mhr_t - mhr_t.mean(0)) @ R.T + (mhr_t.mean(0) * s @ R.T) + t  # equivalent reconstruction
    # The simpler, equivalent expression:
    mhr_aligned = s * (mhr_t @ R.T) + t
    report_aligned_matches(bvh, mhr_aligned, mhr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
