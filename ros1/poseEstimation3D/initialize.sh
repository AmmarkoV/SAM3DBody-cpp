#!/bin/bash
# initialize.sh — fetch the one dependency (AmmarServer, for its minimal
# AmmClient HTTP client) into dependencies/.  The AmmClient C sources are then
# compiled straight into the node by catkin (see CMakeLists.txt) — no prebuilt
# library, no LD_LIBRARY_PATH on the robot.  Mirrors mocapnet_rosnode's flow.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# --- system deps -------------------------------------------------------------
SYSTEM_DEPENDENCIES="git"
for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
  PKG_OK=$(dpkg-query -W --showformat='${Status}\n' "$REQUIRED_PKG" 2>/dev/null | grep "install ok installed")
  echo "Checking for $REQUIRED_PKG: $PKG_OK"
  if [ "" = "$PKG_OK" ]; then
    echo "No $REQUIRED_PKG. Installing.."
    sudo apt-get install $SYSTEM_DEPENDENCIES
    break
  fi
done

mkdir -p dependencies

# --- AmmarServer (AmmClient) -------------------------------------------------
if [ -f dependencies/AmmarServer/src/AmmClient/AmmClient.c ]; then
  echo "AmmarServer (AmmClient) already present."
else
  echo "Cloning AmmarServer (for AmmClient).."
  cd "$DIR/dependencies"
  git clone --depth 1 https://github.com/AmmarkoV/AmmarServer
  cd "$DIR"
fi

if [ -f dependencies/AmmarServer/src/AmmClient/AmmClient.c ]; then
  echo "Done. Now build the workspace:  cd <catkin_ws> && catkin_make"
  echo "Then run:  ./run_it.sh   (or roslaunch poseEstimation3D poseEstimation3D.launch)"
else
  echo "ERROR: AmmClient sources missing after clone — check network/git." >&2
  exit 1
fi

exit 0
