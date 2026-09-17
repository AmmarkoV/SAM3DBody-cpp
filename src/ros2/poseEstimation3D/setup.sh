#!/usr/bin/env bash
# setup.sh — get this package ready to run: build the SAM3DBody-cpp engine it
# drives, fetch the models, and check the handful of Python packages the node
# needs.
#
# Unlike magician_body_pose_estimation there is no virtualenv here: the pose
# engine is a C++ shared library, so the node only needs rclpy (from your ROS 2
# installation), numpy and OpenCV — and a venv would only hide rclpy.
#
# Usage:
#   ./setup.sh                 # build the engine + fetch the default (CUDA) models
#   ./setup.sh --refined       # also fetch the models --refined-pose needs
#   ./setup.sh --skip-build    # models only
#   ./setup.sh --skip-models   # build only
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"   # -P: resolve the workspace symlink
REPO="$( cd "$DIR/../../.." && pwd -P )"

REFINED=0
SKIP_BUILD=0
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --refined)     REFINED=1 ;;
        --skip-build)  SKIP_BUILD=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "=== poseEstimation3D setup ==="
echo "Package:        $DIR"
echo "SAM3DBody-cpp:  $REPO"

if [ ! -f "$REPO/CMakeLists.txt" ]; then
    echo "ERROR: $REPO does not look like a SAM3DBody-cpp checkout."
    echo "This package expects to live at <SAM3DBody-cpp>/src/ros2/poseEstimation3D"
    echo "(symlink it into your ROS 2 workspace rather than copying it)."
    exit 1
fi

# ── 1. the engine ─────────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = "0" ]; then
    echo ""
    echo "--- Building libfast_sam_3dbody.so ---"
    # scripts/setup.sh installs the apt/CUDA dependencies on a fresh machine;
    # once those are in place build.sh is the quick path.
    if [ -f "$REPO/build/CMakeCache.txt" ]; then
        bash "$REPO/scripts/build.sh"
    else
        bash "$REPO/scripts/setup.sh" --skip-venv --skip-models
    fi
fi

# ── 2. the models ─────────────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = "0" ]; then
    echo ""
    echo "--- Fetching models into $REPO/onnx ---"
    if [ "$REFINED" = "1" ]; then
        bash "$REPO/tools/fetch_model.sh" shared cuda refined
    else
        bash "$REPO/tools/fetch_model.sh" shared cuda
    fi
fi

# ── 3. the node's Python dependencies ─────────────────────────────────────────
echo ""
echo "--- Checking Python dependencies ---"
MISSING=""
for module in numpy cv2; do
    python3 -c "import $module" 2>/dev/null || MISSING="$MISSING $module"
done
if [ -n "$MISSING" ]; then
    echo "Missing Python module(s):$MISSING"
    echo "Install them with:  sudo apt install python3-numpy python3-opencv"
else
    echo "numpy and OpenCV are available."
fi
python3 -c "import rclpy" 2>/dev/null \
    || echo "rclpy is not importable — source your ROS 2 setup.bash before running the node."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Build the workspace next:"
echo "  cd <your_ros2_ws> && colcon build --packages-select poseEstimation3D"
echo "  source install/setup.bash"
echo ""
echo "Then run:"
echo "  ros2 run poseEstimation3D poseEstimation3D --input /dev/video0 --display --render"

exit 0
