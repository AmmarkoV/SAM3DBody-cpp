# poseEstimation3D — ROS 1 client for the SAM3DBody-cpp net server

A **minimal, drop-in ROS 1 (catkin) node** that gives an old robot a brand-new
3D human-pose stack with **no GPU and no heavy dependencies on the robot**. It
streams a camera image topic to a remote GPU server
([`sam_3dbody_net`](../../CLIENTSERVER.md)) and republishes the result as a
**TF tree** and a **`std_msgs/Float32MultiArray`**, the same way FORTH's
`mocapnet_rosnode` did — but all the inference (YOLO + DINOv3 backbone + MHR +
forward kinematics) runs on the server.

```
 robot (ROS 1)                                   GPU box
 ────────────                                     ───────
 RGB image topic ─▶ poseEstimation3D ─JPEG POST▶  sam_3dbody_net --server --jlocal
   /camera/.../image_raw     │        ◀─SAM3D txt─       (YOLO+backbone+MHR+FK)
                             ▼
                    TF tree  + /poseEstimation3D/skeleton (Float32MultiArray)
                             + /poseEstimation3D/raw      (std_msgs/String)
```

The robot side only does JPEG encode + TF publish, so it runs comfortably on
old hardware (e.g. a PAL robot stuck on ROS 1).

## Prerequisites

- A ROS 1 catkin workspace (Melodic / Noetic), with `cv_bridge`, `tf2_ros`.
- The GPU server reachable on the network, started with **`--jlocal`**:
  ```bash
  ./build/sam_3dbody_net --server --port 8080 --jlocal
  ```

## Install & build

```bash
# 1. drop this package into your catkin workspace src/
cp -r poseEstimation3D ~/catkin_ws/src/

# 2. fetch the AmmClient dependency (cloned into dependencies/, compiled into the node)
cd ~/catkin_ws/src/poseEstimation3D && ./initialize.sh

# 3. build
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

## Run

```bash
# serverHost and serverPort are required; roslaunch stops if either is missing.
roslaunch poseEstimation3D poseEstimation3D.launch \
    serverHost:=192.168.1.50 serverPort:=8080 fromRGBTopic:=/xtion/rgb/image_raw
# or:  ./run_it.sh serverHost:=192.168.1.50 serverPort:=8080
rviz   # add a TF display to see the skeleton(s)
```

## Parameters

| Param | Default | Meaning |
|---|---|---|
| `serverHost` / `serverPort` | *(required)* | GPU pose server address |
| `fromRGBTopic` | `/camera/rgb/image_raw` | input `sensor_msgs/Image` topic |
| `jpegQuality` | `80` | upload JPEG quality |
| `tfRoot` | `map` | parent of the camera frame |
| `cameraFrame` | `poseEstimation3DCamera` | frame the skeleton hangs off |
| `personFramePrefix` | `p` | per-person frame namespace → `p0/<joint>` |
| `publishCameraTF` | `1` | publish `tfRoot → cameraFrame` |
| `useSimple3DPointTF` | `0` | `0` = oriented `JLOCAL` tree (server `--jlocal`); `1` = positions-only from `KP3D` |
| `cameraXYZPosition` | `0` | camera position in `tfRoot` (m) |
| `cameraRoll/Pitch/Yaw` | `-90 / 0 / -90` | **degrees**; camera-optical → ROS REP-103 orientation |

## Coordinate frames (important)

The server emits each joint transform **parent-relative** in the MHR
**camera-optical** frame (x-right, y-down, z-forward, metres). The whole
skeleton hangs off `cameraFrame`, so a **single** `tfRoot → cameraFrame`
rotation reorients everything into the robot's world — no per-joint fixing.

The default `cameraRoll/Pitch/Yaw = -90 / 0 / -90` is the standard
camera-optical → ROS **REP-103** (x-forward, y-left, z-up) rotation. **If the
skeleton appears rotated or mirrored in RViz, tune these three numbers** (this
is the same workflow `mocapnet_rosnode` exposed). Position the camera in the
world with `cameraXYZPosition`.

## Published topics & TF

- **TF**: `tfRoot → cameraFrame → p<ID>/body_world → p<ID>/root → …` — one
  subtree per tracked person; joint names are the MHR joint names. In
  `useSimple3DPointTF=1` mode: `cameraFrame → p<ID>/kp0…kp69` (positions only).
- **`/poseEstimation3D/skeleton`** (`std_msgs/Float32MultiArray`): flat, per
  person `[ id, njoints, (qx qy qz qw tx ty tz) × njoints ]` (JLOCAL mode).
- **`/poseEstimation3D/raw`** (`std_msgs/String`): the lossless SAM3D text
  response, for any downstream parser.

## Notes

- Single-threaded `ros::spin()` with subscriber queue = 1 ⇒ **drop-latest**:
  while the node blocks on the HTTP round-trip, intermediate frames are dropped,
  keeping it real-time at the server's rate.
- Multiple people get independent TF subtrees (`p0/…`, `p1/…`), keyed by the
  server's tracker id.
- This package depends only on AmmClient (vendored via `initialize.sh`) and
  standard ROS messages — no custom message generation.
