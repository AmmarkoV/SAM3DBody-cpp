# Reference depth map for `tools/hand_depth_error.py`

- `issue15_python_gt_depth.npy` — Python's real SAM-3D-Body depth render for
  `issue15.jpg` (repo root) with the forced bbox below. Generated with `pyrender`
  (offscreen, `fx=fy=focal_length`, `cx=W/2`, `cy=H/2` — the same camera convention
  `mhr_camera_matrices()` / `--save-depth` uses), float32, shape `(H, W)` = `(1382, 1080)`,
  metres, `0` = background. See POSEREFINE.md's "Hand depth-error measurement tool" section
  for the full derivation and how this file was produced (`render_python_gt.py`, run against
  the `sam-3d-body` reference repo — not part of this repo, so this `.npy` is the frozen
  output, not something `tools/hand_depth_error.py` can regenerate on its own).
- `issue15_forced_box.txt` — the external person bbox (`x1 y1 x2 y2`, one line) used for
  both that Python render and any `fast_sam_3dbody_render --boxes` run this reference is
  compared against. The comparison is only meaningful when both sides used this exact box.

Regenerating this file requires the `sam-3d-body` checkpoint + a CUDA GPU + `pyrender`
with EGL — out of scope for this repo. Treat it as a frozen reference snapshot.
