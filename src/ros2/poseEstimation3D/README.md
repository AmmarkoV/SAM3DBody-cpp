# poseEstimation3D — SAM3DBody-cpp ROS 2 node

A ROS 2 package for real-time 3D human body pose estimation using
[SAM3DBody-cpp](../../../README.md). It captures frames from a webcam, detects,
tracks and reconstructs every person with the C++ SAM-3D-Body engine, and
publishes the results as ROS 2 messages and TF transforms.

It is a drop-in replacement for
[`magician_body_pose_estimation`](https://github.com/AmmarkoV/magician_body_pose_estimation):
same topic, same message shape, same joint order, same TF frames, same ArUco
calibration and the same command-line flags — with D-PoSE swapped out for the
SAM-3D-Body engine. Symlink this directory into a ROS 2 workspace and it behaves
like that package did.

The practical difference is what has to be installed: the pose engine is a C++
shared library (`libfast_sam_3dbody.so`) driven through ctypes, so there is **no
torch, no D-PoSE checkout and no virtualenv** — `rclpy`, `numpy` and OpenCV are
the whole Python dependency list.

## Requirements

- ROS 2 Humble (or later)
- CUDA-capable GPU (`--cuda -1` runs on the CPU, slowly)
- Python 3.8+ with numpy and OpenCV (`sudo apt install python3-numpy python3-opencv`)
- A webcam or USB camera

## Setup

Symlink this package into your ROS 2 workspace — do not copy it, since it finds
the engine and the models relative to the SAM3DBody-cpp checkout it lives in:

```bash
ln -s /abs/path/to/SAM3DBody-cpp/src/ros2/poseEstimation3D ~/ros2_ws/src/poseEstimation3D
cd ~/ros2_ws/src/poseEstimation3D && ./setup.sh      # builds the engine, fetches the models
```

`setup.sh` builds `libfast_sam_3dbody.so` in the checkout's `build/`, downloads
the ONNX/GGUF/LBS models into its `onnx/`, and checks the Python modules the
node needs. Add `--refined` to also fetch the models `--refined-pose` uses.

Then build the ROS 2 package:

```bash
cd ~/ros2_ws
colcon build --packages-select poseEstimation3D
source install/setup.bash
```

## Running

```bash
ros2 run poseEstimation3D poseEstimation3D --input /dev/video0 --display --render
```

or run the script in place (this is what the launch script does, and it needs no
`colcon build` except for the messages):

```bash
python3 src/poseEstimation3D/poseEstimation3D_node.py --input /dev/video0
```

Or use the launch script, configured for the Pilot PC with `/dev/video4`:

```bash
bash src/poseEstimation3D/scripts/runROSBodyPoseEstimation.sh
```

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `/dev/video0` | Camera device or video file path (`webcam`, `/dev/videoN`, or file path) |
| `--width` | `1920` | Camera capture width in pixels |
| `--height` | `1080` | Camera capture height in pixels |
| `--fps` | `15` | Camera capture frame rate |
| `--render` | off | Draw the estimated skeleton, projected back into the image, on the OpenCV window (implies `--display`) |
| `--display` | off | Show real-time video window (press `q` to quit) |
| `--refined-pose` | off | Run the iterative refined-pose pass (per-hand decoder + wrist-IK splice): markedly better hands at roughly half the frame rate. Needs the `refined` model profile — `./setup.sh --refined` |
| `--use-aruco` / `--no-use-aruco` | on | Enable ArUco marker detection for camera-to-world calibration. Skeletons are then published w.r.t. the marker |
| `--aruco-marker-id` | any | Only use this marker ID (`DICT_6X6_250`) as the skeleton reference |
| `--aruco-marker-length` | `0.15` | Printed marker side length in meters |
| `--aruco-parent-frame` | `wood_panel` | TF parent frame the marker is mounted on |
| `--aruco-xyz` | `0.1055 1.405 -0.1025` | Marker position in the parent frame (m) |
| `--aruco-rpy` | `90 0 180` | Marker orientation in the parent frame (deg, fixed-axis roll X, pitch Y, yaw Z) |
| `--fx` `--fy` `--cx` `--cy` | `0 0 0 0` (auto) | Camera intrinsics in pixels at `--width` x `--height`. `0` means focal = image diagonal, principal point = image centre — the values SAM-3D-Body was trained with. Used by the pose engine **and** the ArUco pose |
| `--dist-coeffs` | none | Lens distortion coefficients `k1 k2 p1 p2 [k3 ...]` |
| `--static-camera` | off | Camera never moves: average marker observations per slider position, skip detection once converged, and keep publishing through occlusion |
| `--static-robot` | off | Robot never moves along its slider: average everything into one position, no slider topic needed |
| `--slider-topic` | `/slider/position_y` | `std_msgs/Float64` topic with the robot slider position in meters |
| `--slider-bin` | `0.01` | Slider positions this far apart (m) share one averaged marker pose |
| `--marker-table-file` | `<output-folder>/aruco_marker_table.json` | Where learned marker poses are saved on shutdown and reloaded on start |
| `--marker-min-samples` | `30` | Observations before a position is trusted enough to skip detection |
| `--marker-spread-threshold` | `0.01` | Required accuracy (m) of the averaged marker position |
| `--marker-recheck-interval` | `5.0` | How often (s) a converged position is re-detected, to catch a bumped camera |
| `--no-marker-line-fit` | fit on | Never interpolate between slider positions; by default a line is fitted through the learned positions so unvisited ones are predicted |
| `--marker-fit-residual` | `0.02` | Largest residual (m) the rail line fit may have before interpolation is refused |
| `--marker-alarm-pct` | `5.0` | After loading from disk, warn while live observations disagree by more than this percentage of the marker distance |
| `--insist-camera` | off | Retry camera initialization indefinitely (3 s between attempts) |
| `--detection-threshold` | `0.5` | Person detection confidence threshold (0.0–1.0) |
| `--nms` | `0.45` | NMS IoU threshold for person detection |
| `--max-skeletons` | `0` | Maximum people per frame (0 = unlimited) |
| `--cuda` | `0` | CUDA device index for the engine (-1 = CPU) |
| `--lib-dir` | `<checkout>/build` | Directory holding `libfast_sam_3dbody.so` |
| `--onnx-dir` | `<checkout>/onnx` | Directory holding the model files |
| `--gguf` `--yolo` | from `--onnx-dir` | Override individual model paths |
| `--output-folder` | `./logs` | Directory for the ArUco marker table |

Example with common options:

```bash
python3 poseEstimation3D_node.py \
    --input /dev/video0 \
    --render \
    --use-aruco \
    --insist-camera \
    --refined-pose
```

## Exporting Pose Data to CSV

To process a video file and export all detected 3D skeletons to a CSV, use the
provided script:

```bash
bash scripts/bodyPoseEstimationToCSV.sh /path/to/video.mp4
```

This runs the checkout's `python/fast_sam_3dbody_dump_dpose_compat_csv.py` on the
video and writes the output to `<video>_3DBody.csv` alongside the source file.
The joint names and column layout are D-PoSE's, so the CSVs are interchangeable
with the ones `magician_body_pose_estimation` produced:

```
GX010036_out.mp4  →  GX010036_out.mp4_3DBody.csv
```

The CSV contains one row per skeleton per frame, with columns `frame_id`,
`skeleton_id`, and an `x/y/z` triplet for every joint (body, hands, and face
landmarks).

## ROS 2 Interface

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/humans` | `poseEstimation3D/Skeletons` | All detected and tracked skeletons |

### TF Frames

The node publishes a TF transform from `Camera` → `human_<id>` for each tracked
person. The orientation is derived from the pelvis, hips, and neck joints.

When `--use-aruco` is active, additional transforms are published:
`<aruco-parent-frame>` → `Aruco_marker` → `Camera`. The `/humans` joints and the
`human_<id>` frames are then expressed in the `Aruco_marker` frame (X right, Y up
along the printed marker, Z out of the marker), and no skeletons are published
until the marker has been seen.

### Custom Messages

**`Joint3D`** — a single 3D joint position:
```
float64 x
float64 y
float64 z
```

**`Skeleton`** — one tracked person:
```
Joint3D[] joints   # 22 joints in SMPL body order
uint32   id        # Unique tracking ID (persistent across frames)
```

**`Skeletons`** — all people in the scene:
```
Skeleton[] humans
```

### Joint order

`Skeleton.joints` carries the same 22 SMPL body joints D-PoSE published, in the
same order, so consumers that index joints by number are unaffected:

| # | Joint | # | Joint | # | Joint |
|---|---|---|---|---|---|
| 0 | pelvis | 8 | right_ankle | 16 | left_shoulder |
| 1 | left_hip | 9 | spine3 | 17 | right_shoulder |
| 2 | right_hip | 10 | left_foot | 18 | left_elbow |
| 3 | spine1 | 11 | right_foot | 19 | right_elbow |
| 4 | left_knee | 12 | neck | 20 | left_wrist |
| 5 | right_knee | 13 | left_collar | 21 | right_wrist |
| 6 | spine2 | 14 | right_collar | | |
| 7 | left_ankle | 15 | head | | |

They are built from the engine's MHR output by `fsb_engine.py`: most from the 70
MHR keypoints, and pelvis / spine1-3 / head from the full 127-joint MHR skeleton,
which has no keypoint equivalent. Coordinates are metres in the OpenCV camera
frame (x right, y down, z forward), or in the `Aruco_marker` frame when
`--use-aruco` is on.

## Differences from magician_body_pose_estimation

Everything on the ROS side is the same; these are the things that are not:

- **Engine.** SAM-3D-Body (`libfast_sam_3dbody.so`) instead of D-PoSE, with its
  own YOLO person detector. `--cfg`, `--ckpt`, `--detector`, `--yolo-img-size`
  and `--tracker-batch-size` are gone; `--lib-dir`, `--onnx-dir`, `--gguf`,
  `--yolo`, `--cuda`, `--nms`, `--max-skeletons` and `--refined-pose` are new.
- **No venv.** No torch, no D-PoSE clone, no `pip install` — `setup.sh` builds
  the engine and fetches the models instead.
- **Tracking.** A greedy bbox-IoU tracker (`tracker.py`, the same one the
  SAM3DBody-cpp net server uses) replaces SORT, so ids stay consistent with the
  `p_<id>.bvh` files that server writes.
- **`--render`** draws the estimated skeleton projected back into the image,
  rather than a rendered 3D mesh.
- **Intrinsics.** `--fx/--fy/--cx/--cy` default to `0` = auto (focal = image
  diagonal, principal point = image centre) and feed the pose engine as well as
  the ArUco pose. D-PoSE's defaults (`800 800 320 240`) only fed ArUco and were a
  640x480 placeholder; pass them explicitly to reproduce them exactly.
- **`--detection-threshold`** defaults to `0.5` (this detector's own default)
  instead of `0.7`.
- **Logging** goes to the ROS logger only; there is no loguru log file.
  `--output-folder` is still where the ArUco marker table lives.

## Package Structure

```
poseEstimation3D/
├── poseEstimation3D_node.py           # Main ROS 2 node
├── fsb_engine.py                      # ctypes bridge + MHR→SMPL-22 joint mapping
├── tracker.py                         # Greedy bbox-IoU tracker (stable person ids)
├── aruco/                             # ArUco detection + averaged marker-pose table
│   ├── aruco_create.py
│   └── marker_pose_table.py
├── setup.sh                           # Builds the engine, fetches the models
├── scripts/
│   ├── runROSBodyPoseEstimation.sh    # Launch script for the workspace
│   └── bodyPoseEstimationToCSV.sh     # Export video → _3DBody.csv
├── msg/
│   ├── Joint3D.msg
│   ├── Skeleton.msg
│   └── Skeletons.msg
├── CMakeLists.txt
└── package.xml
```

`fsb_engine.py` runs standalone for a quick check that the engine and the joint
mapping work, without ROS:

```bash
python3 fsb_engine.py /path/to/video.mp4 [--refined-pose]
```

## License

MIT — see `package.xml` for maintainer details.
