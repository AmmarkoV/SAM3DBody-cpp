#!/usr/bin/env python3
"""bvh_compare.py — wrap-aware comparison of two sets of BVH files.

Why this exists
---------------
The refined-pose pipeline is NOT bit-reproducible: CUDA GEMMs reassociate
between runs, so two runs of the *same* binary differ slightly.  A refactor is
therefore "correct" when it stays inside that run-to-run spread, not when it
produces byte-identical output.  Comparing against zero produces false alarms;
comparing against the baseline's own noise floor is the real test.

Two details that produced wrong conclusions before:

  * Rotation channels wrap.  A 359.78 vs 0.02 pair is a 0.24 deg difference,
    not 359.76 — so rotation channels are compared modulo 360 and position
    channels linearly.  Which is which is read from the CHANNELS declarations,
    not assumed.
  * `max` over ~500 channels x N frames is a noisy statistic.  A single run
    pair can differ 2x on max while the means are identical, so both are
    reported and the mean is the one to trust for a verdict.

Usage
-----
    tools/bvh_compare.py A B            # compare A_*.bvh against B_*.bvh
    tools/bvh_compare.py A B --max 0.05 # exit 1 if any max exceeds 0.05

Typical use is three runs: baseline vs baseline (the noise floor), then
candidate vs baseline, and check the second is not worse than the first.
"""

import argparse
import glob
import os
import sys


def load(path):
    """-> (is_rotation[nchan], rows).  Channel kinds come from the CHANNELS
    declarations so position vs rotation is never guessed."""
    chans = []
    rows = []
    in_motion = False
    with open(path) as f:
        for line in f:
            t = line.strip()
            if not in_motion:
                if t.startswith("CHANNELS"):
                    # "CHANNELS 6 Xposition Yposition Zposition Zrotation ..."
                    chans += [c.endswith("rotation") for c in t.split()[2:]]
                elif t.startswith("Frame Time"):
                    in_motion = True
                continue
            if not t:
                continue
            vals = [float(x) for x in t.split()]
            if len(vals) == len(chans):
                rows.append(vals)
    return chans, rows


def compare(path_a, path_b):
    """-> (max_diff, mean_diff) or raises ValueError on a shape mismatch."""
    ca, ra = load(path_a)
    cb, rb = load(path_b)
    if ca != cb:
        raise ValueError(f"channel layout differs ({len(ca)} vs {len(cb)})")
    if len(ra) != len(rb):
        raise ValueError(f"frame count differs ({len(ra)} vs {len(rb)})")
    if not ra:
        raise ValueError("no motion rows")

    worst = 0.0
    total = 0.0
    n = 0
    for row_a, row_b in zip(ra, rb):
        for k, (u, v) in enumerate(zip(row_a, row_b)):
            d = abs(u - v)
            if ca[k]:
                d %= 360.0
                if d > 180.0:
                    d = 360.0 - d
            worst = max(worst, d)
            total += d
            n += 1
    return worst, total / n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="first BVH prefix (matches <a>_*.bvh)")
    ap.add_argument("b", help="second BVH prefix")
    ap.add_argument("--max", type=float, default=None,
                    help="fail (exit 1) if any per-person max exceeds this")
    args = ap.parse_args()

    files = sorted(glob.glob(args.a + "_*.bvh"))
    if not files:
        print(f"no files matching {args.a}_*.bvh", file=sys.stderr)
        return 2

    rc = 0
    worst_overall = 0.0
    worst_mean = 0.0
    for fa in files:
        fb = args.b + os.path.basename(fa)[len(os.path.basename(args.a)):]
        name = os.path.basename(fa)
        if not os.path.exists(fb):
            print(f"  {name:24s} MISSING counterpart {os.path.basename(fb)}")
            rc = 1
            continue
        try:
            mx, mean = compare(fa, fb)
        except ValueError as e:
            print(f"  {name:24s} {e}")
            rc = 1
            continue
        worst_overall = max(worst_overall, mx)
        worst_mean = max(worst_mean, mean)
        flag = ""
        if args.max is not None and mx > args.max:
            flag = "  ** EXCEEDS --max **"
            rc = 1
        print(f"  {name:24s} max={mx:.6f}  mean={mean:.6f}{flag}")

    # Both summary numbers matter, and they behave very differently:
    #   mean is stable run-to-run (~+-25%), so a real regression shows up here;
    #   max is a single worst channel out of ~500 x N frames and swings by 3x
    #   between runs of the SAME binary, so it only bounds localised breakage.
    print(f"  {'worst over all persons':24s} max={worst_overall:.6f} mean={worst_mean:.6f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
