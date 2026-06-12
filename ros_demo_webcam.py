#!/usr/bin/env python3
"""
Fast-SAM-3D-Body ROS2 Webcam Demo

Real-time 3D human pose estimation using the C++ SAM-3D-Body backend.
Publishes 70 MHR keypoints per person to the 'humans' ROS2 topic and
broadcasts per-person TF body frames, matching the interface of the
original D-PoSE ros_demo_webcam.py.

Joint ordering in published Skeleton.joints follows MHR70_NAMES (70 joints):
  [0-4]   head keypoints  (nose, eyes, ears)
  [5-20]  body + feet
  [21-41] right hand
  [42-62] left hand
  [63-69] extra (olecranon, cubital-fossa, acromion, neck)

Requirements:
    - ROS2 (Humble or later)
    - libfast_sam_3dbody.so built in fast_sam_3dbody_cpp/build/
    - skeleton_msgs ROS2 package

Usage:
    python3 fast_sam_3dbody_cpp/ros_demo_webcam.py
    python3 fast_sam_3dbody_cpp/ros_demo_webcam.py --from 0 --display
    python3 fast_sam_3dbody_cpp/ros_demo_webcam.py --from video.mp4 --headless
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from skeleton_msgs.msg import Skeletons, Skeleton, Joint3D
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf_transformations

# Optional ArUco support (requires D-PoSE aruco module on sys.path)
_ARUCO_AVAILABLE = False
try:
    from aruco.aruco_create import detect_aruco_from_image
    _ARUCO_AVAILABLE = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# MHR70 joint names  (from sam_3d_body/metadata/mhr70.py original_keypoint_info)
# ──────────────────────────────────────────────────────────────────────────────

MHR70_NAMES = [
    "nose",                        #  0
    "left_eye",                    #  1
    "right_eye",                   #  2
    "left_ear",                    #  3
    "right_ear",                   #  4
    "left_shoulder",               #  5
    "right_shoulder",              #  6
    "left_elbow",                  #  7
    "right_elbow",                 #  8
    "left_hip",                    #  9
    "right_hip",                   # 10
    "left_knee",                   # 11
    "right_knee",                  # 12
    "left_ankle",                  # 13
    "right_ankle",                 # 14
    "left_big_toe_tip",            # 15
    "left_small_toe_tip",          # 16
    "left_heel",                   # 17
    "right_big_toe_tip",           # 18
    "right_small_toe_tip",         # 19
    "right_heel",                  # 20
    "right_thumb_tip",             # 21
    "right_thumb_first_joint",     # 22
    "right_thumb_second_joint",    # 23
    "right_thumb_third_joint",     # 24
    "right_index_tip",             # 25
    "right_index_first_joint",     # 26
    "right_index_second_joint",    # 27
    "right_index_third_joint",     # 28
    "right_middle_tip",            # 29
    "right_middle_first_joint",    # 30
    "right_middle_second_joint",   # 31
    "right_middle_third_joint",    # 32
    "right_ring_tip",              # 33
    "right_ring_first_joint",      # 34
    "right_ring_second_joint",     # 35
    "right_ring_third_joint",      # 36
    "right_pinky_tip",             # 37
    "right_pinky_first_joint",     # 38
    "right_pinky_second_joint",    # 39
    "right_pinky_third_joint",     # 40
    "right_wrist",                 # 41
    "left_thumb_tip",              # 42
    "left_thumb_first_joint",      # 43
    "left_thumb_second_joint",     # 44
    "left_thumb_third_joint",      # 45
    "left_index_tip",              # 46
    "left_index_first_joint",      # 47
    "left_index_second_joint",     # 48
    "left_index_third_joint",      # 49
    "left_middle_tip",             # 50
    "left_middle_first_joint",     # 51
    "left_middle_second_joint",    # 52
    "left_middle_third_joint",     # 53
    "left_ring_tip",               # 54
    "left_ring_first_joint",       # 55
    "left_ring_second_joint",      # 56
    "left_ring_third_joint",       # 57
    "left_pinky_tip",              # 58
    "left_pinky_first_joint",      # 59
    "left_pinky_second_joint",     # 60
    "left_pinky_third_joint",      # 61
    "left_wrist",                  # 62
    "left_olecranon",              # 63
    "right_olecranon",             # 64
    "left_cubital_fossa",          # 65
    "right_cubital_fossa",         # 66
    "left_acromion",               # 67
    "right_acromion",              # 68
    "neck",                        # 69
]

# Indices used to derive the body-frame TF orientation
_IDX_LEFT_HIP  = 9
_IDX_RIGHT_HIP = 10
_IDX_NECK      = 69


# ──────────────────────────────────────────────────────────────────────────────
# ctypes structs  (must match fast_sam_3dbody_capi.h exactly)
# ──────────────────────────────────────────────────────────────────────────────

class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir",        ctypes.c_char_p),
        ("gguf_path",       ctypes.c_char_p),
        ("yolo_path",       ctypes.c_char_p),
        ("cuda_device",     ctypes.c_int),
        ("skip_body_model", ctypes.c_int),
        ("person_thresh",   ctypes.c_float),
        ("person_nms_iou",  ctypes.c_float),
        ("max_persons",     ctypes.c_int),
        ("focal_x",         ctypes.c_float),
        ("focal_y",         ctypes.c_float),
        ("principal_x",     ctypes.c_float),
        ("principal_y",     ctypes.c_float),
    ]


class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox",         ctypes.c_float * 4),
        ("focal_length", ctypes.c_float),
        ("pred_cam_t",   ctypes.c_float * 3),
        ("global_rot",   ctypes.c_float * 3),
        ("body_pose",    ctypes.c_float * 133),
        ("shape",        ctypes.c_float * 45),
        ("scale",        ctypes.c_float * 28),
        ("hand_pose",    ctypes.c_float * 108),
        ("face_params",  ctypes.c_float * 72),
        ("yolo_kps",     ctypes.c_float * 51),
        ("has_yolo_kps", ctypes.c_int),
        ("kps_3d",       ctypes.c_float * 210),
        ("kps_2d",       ctypes.c_float * 140),
        ("has_kps",      ctypes.c_int),
    ]


def load_library(lib_dir: str) -> ctypes.CDLL:
    is_windows = os.name == 'nt'
    lib_name = "fast_sam_3dbody.dll" if is_windows else "libfast_sam_3dbody.so"
    lib_path = os.path.join(lib_dir, lib_name)
    if not os.path.exists(lib_path):
        sys.exit(f"Library not found: {lib_path}\nBuild the project first.")

    # Add the lib directory and dependencies to PATH/LD_LIBRARY_PATH
    if is_windows:
        abs_lib_dir = os.path.abspath(lib_dir)
        # On Windows, we need to add paths to the DLL search path
        os.environ["PATH"] = abs_lib_dir + os.pathsep + os.environ.get("PATH", "")
        # Also try AddDllDirectory for Python 3.8+ if available
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(abs_lib_dir)
            except Exception as e:
                print(f"Warning: could not add DLL directory {abs_lib_dir}: {e}")
    else:
        prev    = os.environ.get("LD_LIBRARY_PATH", "")
        ort_lib = os.path.join(lib_dir, "onnxruntime_dl", "lib")
        os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, [lib_dir, ort_lib, prev]))

    lib = ctypes.CDLL(lib_path)
    lib.fsb_create.restype  = ctypes.c_void_p
    lib.fsb_create.argtypes = []
    lib.fsb_destroy.restype  = None
    lib.fsb_destroy.argtypes = [ctypes.c_void_p]
    lib.fsb_load.restype  = ctypes.c_int
    lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]
    lib.fsb_process_bgr.restype  = ctypes.c_int
    lib.fsb_process_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(FsbResult),
        ctypes.c_int,
    ]
    return lib


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    assert R.shape == (3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ──────────────────────────────────────────────────────────────────────────────

class PoseEstimationNode(Node):
    """
    ROS2 node for real-time MHR70 pose estimation via the C++ SAM-3D-Body backend.

    Publishes:
      /humans  (skeleton_msgs/Skeletons)  — 70 joints per detected person
    Broadcasts TF:
      Camera → human_N                   — body-frame transform per person
      Aruco_marker → Camera              — camera extrinsics (when --use-aruco)
    """

    def __init__(self, args):
        super().__init__('fast_sam_3dbody_node')
        self.args = args

        self.get_logger().info('Initializing Fast-SAM-3D-Body ROS2 node …')

        self.skeleton_publisher = self.create_publisher(Skeletons, 'humans', 10)
        self.tf_broadcaster     = tf2_ros.TransformBroadcaster(self)

        self._initialize_pipeline()
        self._initialize_camera()

        self.first_rvec   = None
        self.first_tvec   = None
        self.frame_number = 0

        if self.args.use_aruco and not _ARUCO_AVAILABLE:
            self.get_logger().warning(
                '--use-aruco requested but aruco module could not be imported; '
                'ArUco detection disabled.'
            )

        self.get_logger().info('Node ready.')

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize_pipeline(self):
        """Load libfast_sam_3dbody.so and initialise the inference pipeline."""
        cpp_dir = os.path.dirname(os.path.abspath(__file__))
        lib_dir = self.args.lib_dir or os.path.join(cpp_dir, 'build')
        onnx    = self.args.onnx_dir or os.path.join(cpp_dir, 'onnx')
        gguf    = self.args.gguf    or os.path.join(onnx, 'pipeline.gguf')
        yolo    = self.args.yolo    or os.path.join(onnx, 'yolo.onnx')

        self.get_logger().info(f'Loading C library from {lib_dir} …')
        self._lib    = load_library(lib_dir)
        self._handle = self._lib.fsb_create()
        if not self._handle:
            raise RuntimeError('fsb_create() returned NULL')

        cfg = FsbConfig(
            onnx_dir        = onnx.encode(),
            gguf_path       = gguf.encode(),
            yolo_path       = yolo.encode(),
            cuda_device     = self.args.cuda,
            skip_body_model = 0,
            person_thresh   = self.args.thresh,
            person_nms_iou  = self.args.nms,
            max_persons     = self.args.max_skeletons,
            focal_x         = self.args.fx,
            focal_y         = self.args.fy,
            principal_x     = self.args.cx,
            principal_y     = self.args.cy,
        )
        if not self._lib.fsb_load(self._handle, ctypes.byref(cfg)):
            self._lib.fsb_destroy(self._handle)
            raise RuntimeError('Pipeline load failed')

        cap = max(self.args.max_skeletons if self.args.max_skeletons > 0 else 32, 32)
        self._results_buf = (FsbResult * cap)()
        self._max_results = cap
        self.get_logger().info('C pipeline ready.')

    def _initialize_camera(self):
        """Open the video capture source."""
        src = self.args.src
        self._cap = cv2.VideoCapture(int(src) if src.isdigit() else src)

        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open input: {src}')

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if src.isdigit():
            self._cap.set(cv2.CAP_PROP_FPS,          self.args.fps)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.args.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)

        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(f'Camera ready: {w}×{h} @ {fps:.0f} fps')

    # ── Per-frame inference ───────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> list:
        """Run C pipeline on one BGR frame; return list of FsbResult."""
        H, W    = frame_bgr.shape[:2]
        bgr_ptr = frame_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        n = self._lib.fsb_process_bgr(
            self._handle, bgr_ptr, W, H, self._results_buf, self._max_results
        )
        return [self._results_buf[i] for i in range(n)]

    # ── ROS publishing ────────────────────────────────────────────────────────

    def publish_skeletons(self, results: list) -> None:
        """
        Publish MHR70 keypoints for all detected persons.

        Each Skeleton carries 70 Joint3D entries in MHR70_NAMES order.
        Coordinates are body-space (metres); the corresponding camera-space
        root position is broadcast as TF transform human_N under Camera.
        """
        msg          = Skeletons()
        msg.humans   = []
        current_time = self.get_clock().now().to_msg()

        for person_idx, result in enumerate(results):
            if not result.has_kps:
                continue

            kps_3d = np.array(result.kps_3d, dtype=np.float64).reshape(70, 3)
            cam_t  = np.array(result.pred_cam_t, dtype=np.float64)

            human    = Skeleton()
            human.id = person_idx
            human.joints = [
                _make_joint3d(kps_3d[k]) for k in range(70)
            ]

            self._publish_human_tf(current_time, person_idx, kps_3d, cam_t)
            msg.humans.append(human)

        self.skeleton_publisher.publish(msg)

    def _publish_human_tf(self, timestamp, human_id: int,
                          kps_3d: np.ndarray, cam_t: np.ndarray) -> None:
        """
        Broadcast TF frame 'human_N' anchored in camera space.

        Origin : midpoint of left_hip (9) and right_hip (10) + cam_t
        Z axis : hip-midpoint → neck (69)   (body upward direction)
        X axis : left_hip (9) → right_hip (10)
        Y axis : right-hand rule
        """
        left_hip  = kps_3d[_IDX_LEFT_HIP]
        right_hip = kps_3d[_IDX_RIGHT_HIP]
        neck      = kps_3d[_IDX_NECK]
        pelvis    = (left_hip + right_hip) * 0.5

        z_axis = neck - pelvis
        nz = np.linalg.norm(z_axis)
        if nz < 1e-6:
            return
        z_axis /= nz

        x_axis = right_hip - left_hip
        nx = np.linalg.norm(x_axis)
        if nx < 1e-6:
            return
        x_axis /= nx

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis  = np.cross(z_axis, y_axis)
        x_axis /= np.linalg.norm(x_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))
        q = tf_transformations.quaternion_from_matrix(
            np.vstack((np.column_stack((R, [0, 0, 0])), [0, 0, 0, 1]))
        )

        root_cam = pelvis + cam_t

        t = TransformStamped()
        t.header.stamp    = timestamp
        t.header.frame_id = 'Camera'
        t.child_frame_id  = f'human_{human_id}'
        t.transform.translation.x = float(root_cam[0])
        t.transform.translation.y = float(root_cam[1])
        t.transform.translation.z = float(root_cam[2])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)

    def publish_aruco_transforms(self, timestamp) -> None:
        """Publish wood_panel→Aruco_marker and Aruco_marker→Camera TF (unchanged)."""
        t = TransformStamped()
        t.header.stamp    = timestamp
        t.header.frame_id = 'wood_panel'
        t.child_frame_id  = 'Aruco_marker'
        t.transform.translation.x =  0.1055
        t.transform.translation.y =  1.405
        t.transform.translation.z = -0.1025
        t.transform.rotation.x =  0.0
        t.transform.rotation.y = -0.7071068
        t.transform.rotation.z = -0.7071068
        t.transform.rotation.w =  0.0
        self.tf_broadcaster.sendTransform(t)

        if self.first_rvec is not None and self.first_tvec is not None:
            R     = cv2.Rodrigues(self.first_rvec)[0]
            R_inv = R.T
            t_inv = -np.dot(R_inv, self.first_tvec.reshape(3))
            quat  = rotmat_to_quat(R_inv)

            t2 = TransformStamped()
            t2.header.stamp    = timestamp
            t2.header.frame_id = 'Aruco_marker'
            t2.child_frame_id  = 'Camera'
            t2.transform.translation.x = float(t_inv[0])
            t2.transform.translation.y = float(t_inv[1])
            t2.transform.translation.z = float(t_inv[2])
            t2.transform.rotation.x = float(quat[1])
            t2.transform.rotation.y = float(quat[2])
            t2.transform.rotation.z = float(quat[3])
            t2.transform.rotation.w = float(quat[0])
            self.tf_broadcaster.sendTransform(t2)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.get_logger().info('Starting … press Q to quit.')

        fps_ema   = 0.0
        fps_alpha = 0.1
        prev_t    = time.perf_counter()

        try:
            while rclpy.ok():
                ok, frame_bgr = self._cap.read()
                if not ok or frame_bgr is None:
                    self.get_logger().error('Failed to capture frame')
                    break

                self.frame_number += 1

                if self.args.use_aruco and _ARUCO_AVAILABLE:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    rvec, tvec = detect_aruco_from_image(frame_rgb)
                    if rvec is not None:
                        self.first_rvec, self.first_tvec = rvec, tvec

                results      = self.process_frame(frame_bgr)
                current_time = self.get_clock().now().to_msg()

                if results:
                    self.publish_skeletons(results)
                    if self.args.use_aruco and _ARUCO_AVAILABLE:
                        self.publish_aruco_transforms(current_time)

                now     = time.perf_counter()
                fps_ema = fps_alpha / (now - prev_t) + (1 - fps_alpha) * fps_ema
                prev_t  = now

                if self.args.display:
                    hud = (f"FPS {fps_ema:.1f}  |  {len(results)} person(s)  "
                           f"|  frame {self.frame_number}")
                    cv2.putText(frame_bgr, hud, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Fast-SAM-3D-Body ROS2', frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info('Quit requested')
                    break

                rclpy.spin_once(self, timeout_sec=0.001)

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted')
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self.get_logger().info('Cleaning up …')
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, '_handle') and self._handle:
            self._lib.fsb_destroy(self._handle)
        self.get_logger().info('Done.')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_joint3d(xyz: np.ndarray) -> Joint3D:
    j = Joint3D()
    j.x = float(xyz[0])
    j.y = float(xyz[1])
    j.z = float(xyz[2])
    return j


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_arguments():
    cpp_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description='Fast-SAM-3D-Body ROS2 node — real-time MHR70 pose to ROS topics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # C library / model paths
    p.add_argument('--lib-dir',  default=os.path.join(cpp_dir, 'build'),
                   help='Directory containing libfast_sam_3dbody.so')
    p.add_argument('--onnx-dir', default=os.path.join(cpp_dir, 'onnx'),
                   help='Directory containing ONNX/GGUF model files')
    p.add_argument('--gguf',     default=None, help='Override path to pipeline.gguf')
    p.add_argument('--yolo',     default=None, help='Override path to yolo.onnx')
    p.add_argument('--cuda',     type=int,   default=0,
                   help='CUDA device index for the C engine (-1 = CPU)')

    # Detection / tracking
    p.add_argument('--thresh',        type=float, default=0.5,
                   help='Person detection confidence threshold')
    p.add_argument('--nms',           type=float, default=0.45,
                   help='NMS IoU threshold')
    p.add_argument('--max-skeletons', type=int,   default=0,
                   help='Max persons per frame (0 = unlimited)')

    # Camera intrinsics (0 = auto-estimate from frame size)
    p.add_argument('--fx', type=float, default=0.0, help='Focal length X')
    p.add_argument('--fy', type=float, default=0.0, help='Focal length Y')
    p.add_argument('--cx', type=float, default=0.0, help='Principal point X')
    p.add_argument('--cy', type=float, default=0.0, help='Principal point Y')

    # Input source
    p.add_argument('--from',   dest='src', default='0',
                   help='Webcam index, video path, or image path')
    p.add_argument('--width',  type=int, default=1280, help='Capture width  (webcam)')
    p.add_argument('--height', type=int, default=720,  help='Capture height (webcam)')
    p.add_argument('--fps',    type=int, default=30,   help='Capture FPS    (webcam)')

    # Output / extras
    p.add_argument('--display',   action='store_true',
                   help='Show OpenCV preview window (press Q to quit)')
    p.add_argument('--use-aruco', action='store_true',
                   help='Enable ArUco marker detection for Camera TF calibration')

    return p.parse_args()


def main():
    args = parse_arguments()

    rclpy.init()
    try:
        node = PoseEstimationNode(args)
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
