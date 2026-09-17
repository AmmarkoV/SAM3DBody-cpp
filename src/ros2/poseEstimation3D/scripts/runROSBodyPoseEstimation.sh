#!/bin/bash
# Launch the node the way the Pilot PC wants it.  Any extra arguments are
# forwarded, e.g.:
#   ./runROSBodyPoseEstimation.sh --input /dev/video0 --refined-pose
#
# Unlike the D-PoSE version there is no virtualenv to activate — the pose engine
# is a shared library, so system python3 + rclpy is all that is needed.

# Do NOT resolve symlinks here: the package is symlinked into <ws>/src, and the
# workspace we want to source is above that symlink, not above the repo.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG="$( cd "$DIR/.." && pwd )"
WS="$( cd "$PKG/../.." && pwd )"

source ~/.bashrc

if [ -f "$WS/install/setup.bash" ]; then
    source "$WS/install/setup.bash"
else
    echo "WARNING: $WS/install/setup.bash not found — build the workspace first:"
    echo "  cd $WS && colcon build --packages-select poseEstimation3D"
fi

#Specific invokation for Pilot PC
python3 "$PKG/poseEstimation3D_node.py" --use-aruco --display --render --insist-camera --input /dev/video4 "$@"

exit 0
