#!/bin/bash
# Export every 3D skeleton in a video to <video>_3DBody.csv, with the same
# joint names and column layout D-PoSE's demo_webcam_csv.py produced — so the
# CSVs magician_body_pose_estimation wrote and the ones written here are
# interchangeable for downstream tooling.
#
# Usage: ./bodyPoseEstimationToCSV.sh /path/to/video.mp4 [extra args]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P )"   # -P: resolve the workspace symlink
REPO="$( cd "$DIR/../../../.." && pwd -P )"

VIDEO="$(realpath "$1")"
shift
CSV_OUT="${VIDEO}_3DBody.csv"
TMP_OUT=$(mktemp -d)

python3 "$REPO/python/fast_sam_3dbody_dump_dpose_compat_csv.py" \
        --input "$VIDEO" --output_folder "$TMP_OUT" "$@"

mv "$TMP_OUT/3DPoints.csv" "$CSV_OUT"
rm -rf "$TMP_OUT"

echo "Saved to: $CSV_OUT"
