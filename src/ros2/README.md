# ros2/ — ROS 2 bridge for SAM3DBody-cpp

This directory holds the **ROS 2 (ament)** integration for SAM3DBody-cpp. Unlike
its [ROS 1 sibling](../ros1/README.md), which is a thin client for a remote
[`sam_3dbody_net`](../../knowledge/CLIENTSERVER.md) server, this package runs the
engine **in-process on the machine with the GPU** — it is the ROS 2 node you put
on a robot (or next to it) that owns both the camera and the GPU.

```
 camera ─▶ poseEstimation3D ────────────────────────────────▶ /humans (Skeletons)
             │  libfast_sam_3dbody.so (ctypes)                 TF: Camera → human_<id>
             │  YOLO + DINOv3 backbone + MHR + LBS                 Aruco_marker → Camera
             └─ ArUco marker → camera extrinsics
```

## Package: `poseEstimation3D/`

A drop-in replacement for
[`magician_body_pose_estimation`](https://github.com/AmmarkoV/magician_body_pose_estimation),
FORTH's D-PoSE-based ROS 2 node: same `/humans` topic, same `Joint3D` /
`Skeleton` / `Skeletons` messages, same 22-joint SMPL ordering, same
`Camera`/`Aruco_marker` → `human_<id>` TF tree, and the same command-line flags —
including the whole ArUco calibration stack (per-slider-position marker-pose
averaging, persistence, rail-line interpolation).

What changes is the engine underneath: the SAM-3D-Body C++ pipeline instead of
D-PoSE. Because that engine is a shared library rather than a Python model, the
node needs **no torch, no D-PoSE checkout and no virtualenv** — `rclpy`, `numpy`
and OpenCV are the entire Python dependency list — and it gained
`--refined-pose` for the iterative hand refinement.

See [`poseEstimation3D/README.md`](poseEstimation3D/README.md) for the full
argument list, the joint order, and the exact differences from the D-PoSE node.

## Install

The package finds the engine and the models relative to the SAM3DBody-cpp
checkout it lives in, so **symlink it** into the workspace rather than copying
it:

```bash
# 1. symlink the package into your ROS 2 workspace src/
ln -s /abs/path/to/SAM3DBody-cpp/src/ros2/poseEstimation3D ~/ros2_ws/src/poseEstimation3D

# 2. build the engine + fetch the models (add --refined for --refined-pose)
cd ~/ros2_ws/src/poseEstimation3D && ./setup.sh

# 3. build the workspace (this generates the three messages)
cd ~/ros2_ws && colcon build --packages-select poseEstimation3D && source install/setup.bash

# 4. run
ros2 run poseEstimation3D poseEstimation3D --input /dev/video0 --display --render
rviz2   # add a TF display to see the skeleton(s)
```

The symlink is also what makes development in place work: colcon builds the live
tree in this repo, so edits land here and can be committed immediately. The
installed copy remembers the checkout it was built from (CMake writes
`engine_root.py` next to the installed node), so `ros2 run` needs no `--lib-dir`.

## Which one do I want?

| | `ros1/poseEstimation3D` | `ros2/poseEstimation3D` |
|---|---|---|
| ROS | 1 (catkin) | 2 (ament) |
| Where inference runs | remote GPU box (`sam_3dbody_net --server`) | in the node |
| GPU on the robot | not needed | required (`--cuda -1` for CPU) |
| Input | an RGB image **topic** | a camera **device** or video file |
| Output | TF tree + `Float32MultiArray` + raw SAM3D text | `/humans` (`Skeletons`) + TF |
| Drop-in for | `mocapnet_rosnode` | `magician_body_pose_estimation` |
