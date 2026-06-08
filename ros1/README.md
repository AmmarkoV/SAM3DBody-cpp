# ros1/ — ROS 1 bridge for SAM3DBody-cpp

This directory holds the **ROS 1 (catkin)** integration for the
[`sam_3dbody_net`](../CLIENTSERVER.md) client/server stack. It exists so an old
robot stuck on ROS 1 (e.g. a PAL robot) can get a modern 3D human-pose tree
**without a GPU and without heavy dependencies on the robot** — all the
inference runs on a remote GPU box.

```
 robot (ROS 1)                                   GPU box
 ────────────                                     ───────
 RGB image topic ─▶ poseEstimation3D ─JPEG POST▶  sam_3dbody_net --server --jlocal
   /camera/.../image_raw     │        ◀─SAM3D txt─       (YOLO+backbone+MHR+FK)
                             ▼
                    TF tree  + /poseEstimation3D/skeleton (Float32MultiArray)
                             + /poseEstimation3D/raw      (std_msgs/String)
```

## Package: `poseEstimation3D/`

A minimal, self-contained catkin node. The robot side only does **JPEG encode +
TF publish**, so it runs comfortably on old hardware.

What it does:

- Subscribes to an RGB image topic, converts it (cv_bridge → BGR), JPEG-encodes
  it, and POSTs it to `sam_3dbody_net --server` using the vendored **AmmClient**
  HTTP client.
- Parses the line-oriented SAM3D text response and republishes it as:
  - a **TF tree** — `tfRoot → cameraFrame → p<id>/<joint>…`, one subtree per
    tracked person (oriented `JLOCAL` joints by default; positions-only `KP3D`
    with `useSimple3DPointTF:=1`);
  - **`/poseEstimation3D/skeleton`** (`std_msgs/Float32MultiArray`);
  - **`/poseEstimation3D/raw`** (`std_msgs/String`, the lossless response).
- Reorients the camera-optical frame into the robot world with a single tunable
  `cameraRoll/Pitch/Yaw` (default `-90/0/-90`, the REP-103 optical rotation).

It only depends on standard ROS messages + AmmClient — **no custom message
generation**. Start the server with `--jlocal` for the default oriented mode.

See [`poseEstimation3D/README.md`](poseEstimation3D/README.md) for parameters,
topics, and coordinate-frame details.

## Install (portable / self-contained)

`initialize.sh` **vendors** the one dependency (AmmClient) as **real file
copies** into `poseEstimation3D/dependencies/AmmClient/` — never symlinks — so
once initialized the package can be tarred and dropped onto any machine and
still build.

```bash
# 1. copy (or symlink — see below) the package into your catkin workspace src/
cp -r ros1/poseEstimation3D ~/catkin_ws/src/

# 2. vendor AmmClient into the package (real copies, no external paths)
cd ~/catkin_ws/src/poseEstimation3D && ./initialize.sh

# 3. build
cd ~/catkin_ws && catkin_make && source devel/setup.bash

# 4. run (point it at your server + camera topic)
roslaunch poseEstimation3D poseEstimation3D.launch \
    serverHost:=192.168.1.50 fromRGBTopic:=/xtion/rgb/image_raw
```

## Developing in-place via a symlink

For development you usually don't want to copy the package into the workspace —
you want catkin to build the live tree in this repo so edits land here and can
be committed immediately. **Symlink the package into your workspace `src/`**:

```bash
ln -s /abs/path/to/SAM3DBody-cpp/ros1/poseEstimation3D ~/catkin_ws/src/poseEstimation3D
cd ~/catkin_ws/src/poseEstimation3D && ./initialize.sh   # vendors AmmClient locally
cd ~/catkin_ws && catkin_make
```

catkin follows the symlink and builds the in-repo sources. This is safe because
**`initialize.sh` copies AmmClient with `cp -L`** (dereferencing symlinks) into
`dependencies/AmmClient/`, so even when the package — or this repo's top-level
`AmmarServer` — is itself a symlink, the build inputs are concrete files. When
sourcing from the SAM3DBody-cpp dev tree it reuses `../../AmmarServer`; otherwise
it shallow-clones AmmarServer once to grab the files.

The vendored `dependencies/AmmClient/` (and any `dependencies/AmmarServer/`
clone) are git-ignored — regenerate or refresh them anytime with
`./initialize.sh` / `./update.sh`.
