#!/usr/bin/env python3
"""Solve every person in an image by driving SAM3DBody with SAM3 instance masks.

scripts/image.sh relies on the built-in YOLO detector, which letterboxes the
frame to 640x640 before detecting.  On a crowd shot that throws away the small
people: DSC_2856.JPG (5568x3712, ~150 people in an auditorium) yields 14 boxes.
SAM3 segments the same frame into 108 person instances.

This script asks a SAM3 gradio server for those instances, turns each one into a
bounding box, and hands the whole set to the renderer via --boxes.  The renderer
then runs its normal per-person crop → backbone → decoder path over them and
composes one image, so every person is solved in a single consistent camera.

Requires the gradio_client venv used by the SAM3 client:
    /home/ammar/Documents/3dParty/sam3/venv/bin/python

Usage:
    sam3_solve.py DSC_2856.JPG
    sam3_solve.py DSC_2856.JPG --server ammar.gr:7860
    sam3_solve.py DSC_2856.JPG --min-box-px 24 --batch 48
    sam3_solve.py DSC_2856.JPG --keep-boxes boxes.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from gradio_client import Client, handle_file

REPO = os.path.dirname(os.path.abspath(__file__))


def segment(image_path, server, prompt, max_side):
    """Return (instances_or_id_map, scale) from the SAM3 server.

    Preferred path: the server returns a JSON list of per-instance boxes
    alongside the encoded image.  Those boxes are exact — they come straight
    off the mask tensors, so they are unaffected by the id channel's 255-id
    cap and by later masks overwriting earlier ones where people overlap.

    Fallback: older deployments return only the image, whose R channel holds
    instance ids.  That path is lossy (see boxes_from_id_channel) and is kept
    only so this script still runs against a server that has not been updated.

    The server OOMs on full-resolution DSLR frames (5568x3712 asks for an 8 GiB
    allocation), so the image is downscaled to max_side first and the returned
    coordinates are scaled back up by the caller.
    """
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    scale = 1.0
    send_path = image_path
    tmp = None
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        small = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, small)
        send_path = tmp.name
        print(f"[sam3] downscaled {w}x{h} -> {small.shape[1]}x{small.shape[0]} for segmentation")

    print(f"[sam3] connecting to http://{server}")
    client = Client(f"http://{server}")
    try:
        result = client.predict(handle_file(send_path), prompt, api_name="/segment")
    finally:
        if tmp:
            os.unlink(tmp.name)

    if isinstance(result, (tuple, list)):
        image_path_out, payload = result[0], result[1]
        instances = payload.get("instances", []) if isinstance(payload, dict) else []
        if instances:
            print(f"[sam3] server returned {len(instances)} instances (exact boxes)")
            return instances, scale
    else:
        image_path_out = result

    print("[sam3] server returned no instance list — falling back to the "
          "lossy id-channel path (update the server for exact boxes)")
    encoded = cv2.imread(image_path_out, cv2.IMREAD_UNCHANGED)
    if encoded is None:
        sys.exit(f"Failed to read mask returned by server: {image_path_out}")
    return encoded[:, :, 2], scale   # OpenCV is BGR, so [:, :, 2] is the id channel


def boxes_from_json(instances, scale, min_box_px, pad_frac, img_w, img_h, min_fill):
    """One box per instance from the server's JSON, in full-resolution pixels."""
    raw = [(i["bbox"][0], i["bbox"][1], i["bbox"][2], i["bbox"][3], i.get("area", 0))
           for i in instances]
    return _finish_boxes(raw, scale, min_box_px, pad_frac, img_w, img_h, min_fill)


def boxes_from_id_channel(ids, scale, min_box_px, pad_frac, img_w, img_h, min_fill):
    """One box per instance id, mapped back to full-resolution pixels.

    The server hands back a *lossy WebP*, so the id channel arrives corrupted:
    compression smears every id value across the frame, and a plain
    ``ids == n`` test collects noise pixels from edge to edge (which produced
    boxes spanning the full image width).  Two steps recover usable instances:
    a median blur to kill isolated speckle, then per id keep only the largest
    connected component and discard the rest as smear.

    This is a workaround, not a fix — the ids that WebP merges outright are
    unrecoverable here.  Have the server return PNG and this becomes exact.
    """
    ids = cv2.medianBlur(ids, 3)
    boxes = []
    dropped = 0
    for inst in np.unique(ids):
        if inst == 0:
            continue                      # 0 is background
        mask = (ids == inst).astype(np.uint8)
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_cc < 2:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[big, cv2.CC_STAT_AREA] < 60:   # pure speckle, no instance left
            dropped += 1
            continue
        bx, by, bw_, bh_ = stats[big, :4]
        boxes.append((bx, by, bx + bw_, by + bh_, int(stats[big, cv2.CC_STAT_AREA])))

    if dropped:
        print(f"[sam3] dropped {dropped} speckle-only instances")
    return _finish_boxes(boxes, scale, min_box_px, pad_frac, img_w, img_h, min_fill)


def _finish_boxes(raw, scale, min_box_px, pad_frac, img_w, img_h, min_fill):
    """Scale segmentation-space boxes to full resolution, pad, clip, filter."""
    boxes = []
    dropped = 0
    degenerate = 0
    for bx1, by1, bx2, by2, area in raw:
        # Drop masks that cover only a sliver of their own bounding box.  SAM3
        # occasionally returns a degenerate instance — on DSC_2856 a 1873x110
        # strip across 92% of the frame at 2% fill.  The crop step squares the
        # box up, so that one becomes a crop of the entire crowd and the model
        # fits a single giant body to it.  Real people fill >= 0.12 of their box
        # even when heavily occluded, so this separates cleanly.
        box_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
        if box_area > 0 and area > 0 and area / box_area < min_fill:
            degenerate += 1
            continue

        x1, x2 = bx1 / scale, bx2 / scale
        y1, y2 = by1 / scale, by2 / scale

        # Pad like a detector box would; the crop step expects some margin
        # around the subject rather than a mask-tight bound.
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * pad_frac; x2 += bw * pad_frac
        y1 -= bh * pad_frac; y2 += bh * pad_frac
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(float(img_w), x2); y2 = min(float(img_h), y2)

        # Below ~24 px the crop is mostly interpolation and the pose comes back
        # as noise, so these cost time without adding a usable person.
        if min(x2 - x1, y2 - y1) < min_box_px:
            dropped += 1
            continue
        boxes.append((x1, y1, x2, y2))

    if degenerate:
        print(f"[sam3] dropped {degenerate} degenerate masks (fill < {min_fill})")
    if dropped:
        print(f"[sam3] dropped {dropped} instances smaller than {min_box_px}px")
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--server", default="ammar.gr:7860")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="Downscale long edge to this before segmenting (server OOMs above)")
    ap.add_argument("--min-box-px", type=int, default=24,
                    help="Skip instances whose padded box is smaller than this")
    ap.add_argument("--pad-frac", type=float, default=0.08,
                    help="Grow each mask-tight box by this fraction per side")
    ap.add_argument("--min-fill", type=float, default=0.10,
                    help="Drop instances whose mask fills less than this fraction of "
                         "its own bounding box. Catches degenerate SAM3 masks (wide "
                         "thin strips) that would otherwise be cropped as one giant "
                         "person. 0 disables.")
    ap.add_argument("--batch", type=int, default=8,
                    help="Solve at most N people per renderer pass (0 = all at once). "
                         "The backbone puts every crop into a single ONNX run, so a "
                         "crowd at once exhausts GPU memory: 182 people asked for a "
                         "1.4 GiB allocation and 32 still OOMed with ~7 GiB free. "
                         "Raise it if the GPU is otherwise idle.")
    ap.add_argument("--out", default=None, help="Output image (default: <stem>_v<ext>)")
    ap.add_argument("--keep-boxes", default=None, help="Also write the box list here")
    ap.add_argument("extra", nargs="*", help="Extra flags forwarded to the renderer")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read image: {args.image}")
    img_h, img_w = img.shape[:2]

    seg, scale = segment(args.image, args.server, args.prompt, args.max_side)
    if isinstance(seg, list):
        boxes = boxes_from_json(seg, scale, args.min_box_px, args.pad_frac,
                                img_w, img_h, args.min_fill)
    else:
        boxes = boxes_from_id_channel(seg, scale, args.min_box_px, args.pad_frac,
                                      img_w, img_h, args.min_fill)
    print(f"[sam3] {len(boxes)} person boxes")
    if not boxes:
        sys.exit("No person instances returned — nothing to solve.")

    stem, ext = os.path.splitext(args.image)
    out_path = args.out or f"{stem}_v{ext}"

    # One renderer pass per batch.  Each pass re-renders the same source image
    # with a different slice of the boxes, so passes after the first read back
    # the previous output and accumulate onto it.
    chunk = args.batch if args.batch > 0 else len(boxes)
    passes = [boxes[i:i + chunk] for i in range(0, len(boxes), chunk)]
    source = args.image

    with tempfile.TemporaryDirectory(prefix="sam3_solve_") as tmpdir:
        for n, group in enumerate(passes):
            boxes_file = os.path.join(tmpdir, f"boxes_{n}.txt")
            with open(boxes_file, "w") as fh:
                fh.write("# x1 y1 x2 y2 (original image pixels)\n")
                for b in group:
                    fh.write("%.1f %.1f %.1f %.1f\n" % b)
            if args.keep_boxes and n == 0:
                shutil.copy(boxes_file, args.keep_boxes)

            prefix = os.path.join(tmpdir, f"frame_{n}_")
            cmd = [
                os.path.join(REPO, "build", "fast_sam_3dbody_render"),
                "--from", source,
                "--onnx-dir", os.path.join(REPO, "onnx"),
                "--gguf", os.path.join(REPO, "onnx", "pipeline.gguf"),
                "--yolo", os.path.join(REPO, "onnx", "yolo.onnx"),
                "--mesh", os.path.join(REPO, "body_mesh.tri"),
                "--lbs", os.path.join(REPO, "onnx", "body_model.lbs"),
                "--boxes", boxes_file,
                "--render-size", str(img_w), str(img_h),
                "--headless", "--save-frames", prefix,
            ] + args.extra

            if len(passes) > 1:
                print(f"[solve] pass {n + 1}/{len(passes)} — {len(group)} people")
            if subprocess.run(cmd, cwd=REPO).returncode != 0:
                sys.exit("Renderer failed")

            frames = sorted(f for f in os.listdir(tmpdir) if f.startswith(f"frame_{n}_"))
            if not frames:
                sys.exit("Renderer produced no frame")
            rendered = os.path.join(tmpdir, frames[0])

            # Feed this pass's render in as the next pass's background so the
            # meshes stack up into one image.
            source = os.path.join(tmpdir, f"stage_{n}.jpg")
            os.rename(rendered, source)

        shutil.copy(source, out_path)

    print(f"Wrote: {out_path}  ({len(boxes)} people)")


if __name__ == "__main__":
    main()
