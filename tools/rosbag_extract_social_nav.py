#!/usr/bin/env python3
"""
Extract pedestrian position / body heading / head heading from a rosbag that
contains /poseEstimation3D/raw (the lossless SAM3D text topic published by the
src/ros1/poseEstimation3D node).

Everything is derived from the KP3D keypoints, which are root-relative metres in
the camera-optical frame (x-right, y-down, z-forward); absolute camera-space
position is KP3D + CAMT.  Output is in ROS REP-103 (x-forward, y-left, z-up),
using the same remap as the node's default cameraRoll/Pitch/Yaw = -90/0/-90.

Usage:
    python3 tools/rosbag_extract_social_nav.py my.bag -o tracks.csv
    python3 tools/rosbag_extract_social_nav.py --selftest
"""

import argparse
import csv
import math
import sys

# MHR70 keypoint indices (see MHR70_NAMES in ros_demo_webcam.py)
L_HIP, R_HIP, NECK = 9, 10, 69
NOSE, L_EAR, R_EAR = 0, 3, 4


def parse_sam3d(text):
    """SAM3D response text -> [{'id':int, 'camt':[3], 'kp3d':[[x,y,z]*70]}]."""
    persons = []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "P":
            pid = int(tok[2].split("=", 1)[1])
            persons.append({"id": pid, "camt": None, "kp3d": None})
        elif not persons:
            continue
        elif tok[0] == "CAMT":
            persons[-1]["camt"] = [float(v) for v in tok[1:4]]
        elif tok[0] == "KP3D":
            persons[-1]["kp3d"] = [[float(c) for c in p.split(",")] for p in tok[2:]]
    return persons


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return None if n < 1e-9 else [v[0] / n, v[1] / n, v[2] / n]


def to_ros(v):
    """Camera-optical (x-right, y-down, z-forward) -> REP-103 (fwd, left, up).

    Identical to the node's RPY(-90,0,-90) camera transform, so these numbers
    agree with the TF tree the node broadcasts."""
    return [v[2], -v[0], -v[1]]


def person_state(p):
    """-> (x, y, z, body_yaw, head_yaw) in REP-103, or None if unusable.

    body_yaw : ground-plane heading of the torso.  Forward = up x lateral, with
               up = pelvis->neck and lateral = left_hip->right_hip, so it stays
               defined even when the person leans.
    head_yaw : ground-plane heading of the gaze, from the ear midpoint to the
               nose.  Keypoint-derived on purpose: it needs no assumption about
               the MHR rest-pose axis convention baked into the joint rotations.
    """
    kp, camt = p.get("kp3d"), p.get("camt")
    if not kp or len(kp) < 70 or not camt:
        return None

    pelvis = [(kp[L_HIP][i] + kp[R_HIP][i]) * 0.5 for i in range(3)]
    up = _norm(_sub(kp[NECK], pelvis))
    lateral = _norm(_sub(kp[R_HIP], kp[L_HIP]))
    if up is None or lateral is None:
        return None

    pos = to_ros([pelvis[i] + camt[i] for i in range(3)])
    fwd = to_ros(_cross(up, lateral))
    body_yaw = math.atan2(fwd[1], fwd[0])

    ear_mid = [(kp[L_EAR][i] + kp[R_EAR][i]) * 0.5 for i in range(3)]
    gaze = _norm(_sub(kp[NOSE], ear_mid))
    head_yaw = ""
    if gaze is not None:
        g = to_ros(gaze)
        head_yaw = math.atan2(g[1], g[0])

    return pos[0], pos[1], pos[2], body_yaw, head_yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", nargs="?", help="input .bag")
    ap.add_argument("-t", "--topic", default="/poseEstimation3D/raw")
    ap.add_argument("-o", "--out", default="tracks.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.bag:
        ap.error("bag is required (or use --selftest)")

    import rosbag  # ROS 1; source your catkin workspace first

    rows = 0
    with rosbag.Bag(args.bag) as bag, open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "id", "x", "y", "z", "body_yaw", "head_yaw"])
        # The node POSTs /infer with no seq/t_ms query params, so every response
        # carries frame=0 t_ms=0 -- the bag receipt time below is the only usable
        # stamp.  It lags capture by one inference round trip (~200 ms).
        for _, msg, t in bag.read_messages(topics=[args.topic]):
            for p in parse_sam3d(msg.data):
                st = person_state(p)
                if st:
                    w.writerow([t.to_sec(), p["id"]] + list(st))
                    rows += 1
    print("wrote %d rows to %s" % (rows, args.out))


def selftest():
    """Synthetic upright person facing +z_cam (away from the camera, REP-103 +x).

    Facing away means the person's left side is on the IMAGE left, so
    left_hip.x = -0.1 and right_hip.x = +0.1 in the camera-optical frame."""
    text = (
        "SAM3D 1 frame=0 t_ms=0 persons=1\n"
        "P 0 id=7 bbox=1,2,3,4 focal=640\n"
        "GROT 0 0 0\n"
        "CAMT 0.5 0 4\n"
        "KP3D 70 " + " ".join(
            # y is DOWN: hips at y=0, neck above at y=-0.5
            {L_HIP: "-0.1,0,0", R_HIP: "0.1,0,0", NECK: "0,-0.5,0",
             NOSE: "0,-0.6,0.1", L_EAR: "-0.07,-0.6,0", R_EAR: "0.07,-0.6,0"}
            .get(i, "0,0,0") for i in range(70)) + "\n"
        "END frame=0\n")

    persons = parse_sam3d(text)
    assert len(persons) == 1 and persons[0]["id"] == 7, persons
    assert len(persons[0]["kp3d"]) == 70

    x, y, z, byaw, hyaw = person_state(persons[0])
    # pelvis (0,0,0) + camt (0.5,0,4) -> REP-103 (4, -0.5, 0)
    assert abs(x - 4.0) < 1e-6 and abs(y + 0.5) < 1e-6 and abs(z) < 1e-6, (x, y, z)
    # facing the camera axis -> yaw 0 for both body and head
    assert abs(byaw) < 1e-6, byaw
    assert abs(hyaw) < 1e-6, hyaw

    # Turn the person 90 deg (right hip toward +z_cam) -> heading +90 deg (left)
    persons[0]["kp3d"][L_HIP] = [0, 0, -0.1]
    persons[0]["kp3d"][R_HIP] = [0, 0, 0.1]
    byaw = person_state(persons[0])[3]
    assert abs(byaw - math.pi / 2) < 1e-6, byaw

    print("selftest OK")


if __name__ == "__main__":
    sys.exit(main())
