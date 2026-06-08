#!/bin/bash
# initialize.sh — vendor the one dependency (AmmClient, AmmarServer's minimal
# HTTP client) into dependencies/AmmClient/ as REAL FILE COPIES (never symlinks),
# so this package is fully self-contained and portable: tar it, drop it into any
# catkin workspace on any machine, and it builds with no external paths.
#
# Source priority:
#   1. the SAM3DBody-cpp dev tree     (../../AmmarServer, which may be a symlink)
#   2. a previous clone               (dependencies/AmmarServer)
#   3. a fresh shallow git clone of AmmarServer
# The AmmClient .c sources are then compiled straight into the node by catkin
# (see CMakeLists.txt) — no prebuilt library, no LD_LIBRARY_PATH on the robot.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# files needed to build AmmClient (they only #include each other)
AMMCLIENT_FILES="AmmClient.c AmmClient.h network.c network.h protocol.c protocol.h tools.c tools.h"

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

# --- locate a source for the AmmClient files ---------------------------------
SRC=""
if [ -f "$DIR/../../AmmarServer/src/AmmClient/AmmClient.c" ]; then
  SRC="$DIR/../../AmmarServer/src/AmmClient"
  echo "Using AmmClient from the SAM3DBody-cpp dev tree: $SRC"
elif [ -f "$DIR/dependencies/AmmarServer/src/AmmClient/AmmClient.c" ]; then
  SRC="$DIR/dependencies/AmmarServer/src/AmmClient"
  echo "Using AmmClient from a previous clone: $SRC"
else
  echo "Cloning AmmarServer (for AmmClient).."
  git clone --depth 1 https://github.com/AmmarkoV/AmmarServer "$DIR/dependencies/AmmarServer"
  SRC="$DIR/dependencies/AmmarServer/src/AmmClient"
fi

if [ ! -f "$SRC/AmmClient.c" ]; then
  echo "ERROR: could not find AmmClient sources at $SRC — check network/git." >&2
  exit 1
fi

# --- vendor: copy REAL files (dereference symlinks) into dependencies/AmmClient
echo "Vendoring AmmClient into dependencies/AmmClient/ .."
mkdir -p "$DIR/dependencies/AmmClient"
for f in $AMMCLIENT_FILES; do
  if ! cp -Lf "$SRC/$f" "$DIR/dependencies/AmmClient/$f"; then
    echo "ERROR: failed to copy $f from $SRC" >&2
    exit 1
  fi
done

if [ -f "$DIR/dependencies/AmmClient/AmmClient.c" ]; then
  echo "Done. AmmClient vendored (self-contained, no symlinks)."
  echo "Now build the workspace:  cd <catkin_ws> && catkin_make"
  echo "Then run:  ./run_it.sh   (or roslaunch poseEstimation3D poseEstimation3D.launch)"
else
  echo "ERROR: vendoring failed — dependencies/AmmClient/AmmClient.c missing." >&2
  exit 1
fi

exit 0
