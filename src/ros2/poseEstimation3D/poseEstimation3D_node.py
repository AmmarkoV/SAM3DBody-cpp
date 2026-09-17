#!/usr/bin/env python3
"""
poseEstimation3D — SAM3DBody-cpp ROS 2 node

Real-time 3D human pose estimation from a camera, published as ROS 2 skeleton
messages and TF transforms.  Interface-compatible with
magician_body_pose_estimation (same topic, same message shape, same TF frames,
same ArUco calibration), with the SAM-3D-Body C++ engine
(libfast_sam_3dbody.so) in place of D-PoSE — so no torch, no D-PoSE checkout and
no virtualenv on the robot: rclpy, numpy and OpenCV are the only Python
dependencies.

Requirements:
    - ROS 2 (Humble or later)
    - CUDA-capable GPU (--cuda -1 runs on the CPU, slowly)
    - libfast_sam_3dbody.so + the onnx/ models  (bash setup.sh)
    - ArUco marker for camera calibration (optional)

Usage:
    python3 poseEstimation3D_node.py [--input /dev/video0] [--display]
    ros2 run poseEstimation3D poseEstimation3D --input /dev/video0

Author: Ammar Qammaz
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

# Local modules (the node's own directory), imported before ROS so an engine
# problem is reported without a ROS stack in the way.
from aruco.aruco_create import detect_aruco_from_image
from aruco.marker_pose_table import MarkerPoseTable, install_exit_hooks
from fsb_engine import (Engine, SMPL22_NAMES, SMPL22_PARENTS, repo_root_from,
                        smpl22_joints)
from tracker import IoUTracker

# ROS 2 imports
import rclpy
from rclpy.node import Node
from poseEstimation3D.msg import Skeletons, Skeleton, Joint3D
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
import tf2_ros


def getCaptureDeviceFromPath(videoFilePath, videoWidth, videoHeight, videoFramerate=30):
    if videoFilePath == 'webcam' or videoFilePath == '/dev/video0':
        cap = cv2.VideoCapture(0)
    elif videoFilePath.startswith('/dev/video'):
        idx = int(videoFilePath.replace('/dev/video', ''))
        cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(videoFilePath)
    if hasattr(cap, 'set'):
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
    return cap


def rotmat_to_quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).

    Args:
        R (np.ndarray): 3x3 rotation matrix

    Returns:
        np.ndarray: Quaternion as [w, x, y, z]
    """
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix"
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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


def degrees_to_radians(deg):
    """Convert degrees to radians."""
    return math.radians(deg)


class PoseEstimationNode(Node):
    """
    ROS 2 node for real-time 3D human pose estimation using SAM3DBody-cpp.

    Captures frames from a camera, runs the C++ SAM-3D-Body pipeline, and
    publishes the results as skeleton messages and TF transforms.
    """

    def __init__(self, args):
        """
        Initialize the pose estimation node.

        Args:
            args: Command line arguments containing configuration
        """
        super().__init__('poseEstimation3D')

        self.args = args
        self.get_logger().info('Initializing poseEstimation3D node...')

        # Initialize publishers and broadcasters
        self.skeleton_publisher = self.create_publisher(Skeletons, 'humans', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Identity across frames (the engine reports detections, not tracks)
        self.tracker = IoUTracker()
        self._last_no_detection_log = 0.0  # throttle "no human" warnings to once per 2 s
        self._last_stats_log = 0.0
        self._detection_was_active = False   # tracks state changes (detected → lost)
        self._frames_with_detection = 0
        self._frames_total = 0
        self._no_body_model_warned = False

        # Initialize pose estimation engine
        self._initialize_engine()

        # Initialize camera
        self._initialize_camera()

        # Effective camera intrinsics at --width x --height.  0 means "auto",
        # and auto has to be resolved here because the ArUco solvePnP needs real
        # numbers; these are the same values the engine derives internally
        # (focal = image diagonal, principal point = image centre).
        diagonal = math.sqrt(self.args.width ** 2 + self.args.height ** 2)
        self.fx = self.args.fx if self.args.fx > 0 else diagonal
        self.fy = self.args.fy if self.args.fy > 0 else diagonal
        self.cx = self.args.cx if self.args.cx > 0 else self.args.width * 0.5
        self.cy = self.args.cy if self.args.cy > 0 else self.args.height * 0.5
        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx:.1f} cy={self.cy:.1f}'
            f'{" (auto)" if self.args.fx <= 0 else ""}')

        # ArUco detection state
        self.first_rvec = None
        self.first_tvec = None
        self._initialize_marker_table()

        self.get_logger().info('poseEstimation3D node initialized successfully!')

    def _initialize_engine(self):
        """Load libfast_sam_3dbody.so and the models."""
        try:
            self.get_logger().info(
                f'Loading SAM3DBody-cpp engine from {self.args.lib_dir} '
                f'(models: {self.args.onnx_dir})...')
            if self.args.refined_pose:
                self.get_logger().info(
                    'Refined pose is ON: the iterative decoder + wrist-IK pass '
                    'runs on every frame (better hands, roughly half the frame rate)')
            self.engine = Engine(
                lib_dir        = self.args.lib_dir,
                onnx_dir       = self.args.onnx_dir,
                gguf_path      = self.args.gguf,
                yolo_path      = self.args.yolo,
                cuda_device    = self.args.cuda,
                person_thresh  = self.args.detection_threshold,
                person_nms_iou = self.args.nms,
                max_persons    = self.args.max_skeletons,
                fx = self.args.fx, fy = self.args.fy,
                cx = self.args.cx, cy = self.args.cy,
                refined_pose   = self.args.refined_pose,
            )
            self.get_logger().info('SAM3DBody-cpp engine loaded successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to load the engine: {e}')
            raise

    def _try_open_camera(self):
        """Attempt a single camera open and test-grab a frame; return True on success."""
        self.cap = getCaptureDeviceFromPath(
            self.args.input, self.args.width, self.args.height, self.args.fps
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Re-apply resolution after FOURCC — setting MJPG can reset camera to a default resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        if not self.cap.isOpened():
            return False
        ret, _ = self.cap.read()
        return ret

    def _initialize_camera(self):
        """Initialize camera capture, retrying indefinitely when --insist-camera is set."""
        retry_delay = 3.0
        attempt = 0
        while True:
            attempt += 1
            try:
                self.get_logger().info(
                    f'Initializing camera {self.args.input} (attempt {attempt})...'
                )
                if self._try_open_camera():
                    actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.get_logger().info(
                        f'Camera initialized: {int(actual_width)}x{int(actual_height)} @ {actual_fps} FPS'
                    )
                    return
                raise RuntimeError(f"Cannot open camera {self.args.input}")
            except Exception as e:
                self.get_logger().error(f'Failed to initialize camera: {e}')
                if not self.args.insist_camera:
                    raise
                self.get_logger().info(
                    f'--insist-camera: retrying in {retry_delay:.0f}s...'
                )
                time.sleep(retry_delay)

    def process_frame(self, frame_bgr):
        """
        Process a single frame for pose estimation.

        Args:
            frame_bgr (np.ndarray): Input frame in BGR format

        Returns:
            tuple: (track_ids, results) or (None, None) if nobody was detected
        """
        results = self.engine.process(frame_bgr)

        # A detection whose body model did not run carries no 3D joints.
        usable = [r for r in results if r.has_kps and r.has_skel]
        if len(usable) != len(results) and not self._no_body_model_warned:
            self.get_logger().warning(
                'The engine returned detections without body-model output; '
                'those persons are skipped. Check that onnx/body_model.lbs and '
                'onnx/keypoint_mapping.bin are present.'
            )
            self._no_body_model_warned = True

        self._frames_total += 1

        if usable:
            self._frames_with_detection += 1
            if not self._detection_was_active:
                self.get_logger().info(
                    f'Human detected (frame {self._frames_total}). '
                    f'Detection rate so far: '
                    f'{self._frames_with_detection}/{self._frames_total} frames.'
                )
                self._detection_was_active = True
            track_ids = self.tracker.assign([list(r.bbox) for r in usable])
            return track_ids, usable

        if self._detection_was_active:
            self.get_logger().warning(
                f'Human tracking lost (frame {self._frames_total}). '
                f'Detection rate so far: '
                f'{self._frames_with_detection}/{self._frames_total} frames.'
            )
            self._detection_was_active = False
        now = time.time()
        if now - self._last_no_detection_log >= 2.0:
            self.get_logger().warning(
                'No human detected — skipping pose estimation. '
                'Check lighting, camera angle, or lower --detection-threshold '
                f'(currently {self.args.detection_threshold}).'
            )
            self._last_no_detection_log = now
        if now - self._last_stats_log >= 30.0:
            self.get_logger().info(
                f'Detection rate (last 30 s window): '
                f'{self._frames_with_detection}/{self._frames_total} frames had a human.'
            )
            self._last_stats_log = now

        return None, None

    def publish_skeletons(self, track_ids, results):
        """
        Publish skeleton data to ROS topic.

        Args:
            track_ids: Per-person tracking ids
            results: Per-person engine output

        Returns:
            list: the per-person SMPL-22 joints in the camera frame (for the overlay)
        """
        skeletons_msg = Skeletons()
        skeletons_msg.humans = []

        current_time = self.get_clock().now().to_msg()

        marker_R = None
        marker_t = None
        if self.args.use_aruco:
            if self.first_rvec is None or self.first_tvec is None:
                self.get_logger().warning(
                    'No ArUco marker pose yet, not publishing skeletons',
                    throttle_duration_sec=2.0,
                )
                return []
            marker_R = cv2.Rodrigues(self.first_rvec)[0]
            marker_t = self.first_tvec.reshape(3)

        camera_joints = []
        for track_id, result in zip(track_ids, results):
            human = Skeleton()
            human.id = int(track_id)
            human.joints = []

            # Joints of this person in the camera frame
            # (OpenCV axes: x right, y down, z forward)
            joints = smpl22_joints(result)
            camera_joints.append(joints)

            # Express joints w.r.t. the ArUco marker: p_marker = R^T (p_camera - t)
            if marker_R is not None:
                joints = (joints - marker_t) @ marker_R

            # Convert joints to ROS message format
            for joint in joints:
                joint3d = Joint3D()
                joint3d.x = float(joint[0])
                joint3d.y = float(joint[1])
                joint3d.z = float(joint[2])
                human.joints.append(joint3d)

            self._publish_human_camera_transform(current_time, human.id, human)
            skeletons_msg.humans.append(human)

        # Publish the skeleton message
        self.skeleton_publisher.publish(skeletons_msg)
        return camera_joints

    def _publish_human_camera_transform(self, timestamp, human_id, human):
        """
        Publish the TF transform of one person's body frame.

        Args:
            timestamp: ROS timestamp
            human_id: Human ID
            human: the Skeleton message being published
        """
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = 'Aruco_marker' if self.args.use_aruco else 'Camera'
        t.child_frame_id = f'human_{human_id}'

        t.transform.translation.x = human.joints[0].x
        t.transform.translation.y = human.joints[0].y
        t.transform.translation.z = human.joints[0].z

        pelvis = np.array([human.joints[0].x, human.joints[0].y, human.joints[0].z])
        left_hip = np.array([human.joints[1].x, human.joints[1].y, human.joints[1].z])
        right_hip = np.array([human.joints[2].x, human.joints[2].y, human.joints[2].z])
        neck = np.array([human.joints[12].x, human.joints[12].y, human.joints[12].z])

        z_axis = neck-pelvis
        z_axis /= np.linalg.norm(z_axis)

        x_axis = right_hip - left_hip
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))

        q = rotmat_to_quat(R)  # returns (w, x, y, z)

        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])

        self.tf_broadcaster.sendTransform(t)

    def _initialize_marker_table(self):
        """Set up the averaged marker poses and the slider subscription."""
        table_file = self.args.marker_table_file or os.path.join(
            self.args.output_folder, 'aruco_marker_table.json')
        self.marker_table = MarkerPoseTable(
            static_camera=self.args.static_camera,
            static_robot=self.args.static_robot,
            bin_size=self.args.slider_bin,
            min_samples=self.args.marker_min_samples,
            spread_threshold=self.args.marker_spread_threshold,
            recheck_interval=self.args.marker_recheck_interval,
            alarm_pct=self.args.marker_alarm_pct,
            line_fit=self.args.marker_line_fit,
            fit_residual=self.args.marker_fit_residual,
            table_file=table_file,
            logger=self.get_logger(),
        )

        if not (self.args.use_aruco and self.args.static_camera):
            return

        install_exit_hooks(self.marker_table)
        if self.args.static_robot:
            self.get_logger().info('Static robot: averaging all marker observations '
                                   'into a single slider position')
        else:
            self.slider_subscription = self.create_subscription(
                Float64, self.args.slider_topic,
                lambda msg: self.marker_table.set_slider(float(msg.data)), 10)
            self.get_logger().info(
                f'Averaging marker poses per slider position from {self.args.slider_topic}')

    # TODO: Move this to the robot side (so no fixed envirnoment components are present here)
    def publish_aruco_transforms(self, timestamp):
        """
        Publish ArUco marker transforms.

        Args:
            timestamp: ROS timestamp
        """
        # Publish base ArUco transform
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.args.aruco_parent_frame
        t.child_frame_id = 'Aruco_marker'

        t.transform.translation.x = self.args.aruco_xyz[0]
        t.transform.translation.y = self.args.aruco_xyz[1]
        t.transform.translation.z = self.args.aruco_xyz[2]

        # Fixed-axis roll (X), pitch (Y), yaw (Z): R = Rz(yaw) Ry(pitch) Rx(roll)
        roll, pitch, yaw = (degrees_to_radians(a) for a in self.args.aruco_rpy)
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        q = rotmat_to_quat(Rz @ Ry @ Rx)  # returns (w, x, y, z)
        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])

        self.tf_broadcaster.sendTransform(t)

        # Publish camera transform relative to ArUco marker
        if self.first_rvec is not None and self.first_tvec is not None:
            t = TransformStamped()
            t.header.stamp = timestamp
            t.header.frame_id = 'Aruco_marker'
            t.child_frame_id = 'Camera'

            # Convert rotation vector to rotation matrix and invert
            R = cv2.Rodrigues(self.first_rvec)[0]
            R_inv = R.T

            # Invert translation
            tvec = self.first_tvec.reshape(3)
            t_inv = -np.dot(R_inv, tvec)

            t.transform.translation.x = float(t_inv[0])
            t.transform.translation.y = float(t_inv[1])
            t.transform.translation.z = float(t_inv[2])

            # Convert rotation matrix to quaternion
            quat = rotmat_to_quat(R_inv)
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])
            t.transform.rotation.w = float(quat[0])

            self.tf_broadcaster.sendTransform(t)

    def _draw_overlay(self, frame_bgr, track_ids, results, camera_joints):
        """Draw the projected skeleton (--render) on the display frame."""
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        for track_id, result, joints in zip(track_ids, results, camera_joints):
            z = np.maximum(joints[:, 2], 1e-4)
            u = (joints[:, 0] / z * fx + cx).astype(int)
            v = (joints[:, 1] / z * fy + cy).astype(int)

            for child, parent in enumerate(SMPL22_PARENTS):
                if parent >= 0:
                    cv2.line(frame_bgr, (u[parent], v[parent]), (u[child], v[child]),
                             (0, 255, 0), 2, cv2.LINE_AA)
            for k in range(len(SMPL22_NAMES)):
                cv2.circle(frame_bgr, (u[k], v[k]), 3, (0, 255, 255), -1, cv2.LINE_AA)

            x1, y1, x2, y2 = (int(v) for v in result.bbox)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 128, 0), 1)
            cv2.putText(frame_bgr, f'human_{track_id}', (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA)

    def run(self):
        """
        Main processing loop.
        """
        self.get_logger().info('Starting pose estimation loop...')
        if self.args.display or self.args.render:
            self.get_logger().info('Press "q" in the OpenCV window to quit')
        else:
            self.get_logger().info(
                'No OpenCV window: pass --display (and --render for the '
                'projected skeleton) to see what the camera sees')

        try:
            while rclpy.ok():
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error('Failed to capture frame from camera')
                    break

                if frame.shape[1] != self.args.width or frame.shape[0] != self.args.height:
                    self.get_logger().warning(
                        f'Frame size mismatch: camera delivered {frame.shape[1]}x{frame.shape[0]} '
                        f'but {self.args.width}x{self.args.height} was requested — resizing. '
                        'This may reduce detection accuracy.'
                    )
                    frame = cv2.resize(frame, (self.args.width, self.args.height))

                # Process frame for pose estimation.  Done before ArUco detection
                # because the detector draws the marker outline and axes onto the
                # frame it is given, and the network should see clean pixels.
                track_ids, results = self.process_frame(frame)

                # Detect ArUco markers (for camera calibration)
                if self.args.use_aruco:
                    if self.marker_table.should_detect():
                        rvec, tvec = detect_aruco_from_image(
                            frame,
                            marker_id=self.args.aruco_marker_id,
                            fx=self.fx, fy=self.fy,
                            cx=self.cx, cy=self.cy,
                            dist_coeffs=self.args.dist_coeffs,
                            marker_length=self.args.aruco_marker_length,
                            input_is_bgr=True,
                        )
                        if rvec is not None and tvec is not None:
                            self.marker_table.add(rvec, tvec)
                    estimate = self.marker_table.estimate()
                    if estimate is not None:
                        self.first_rvec, self.first_tvec = estimate

                current_time = self.get_clock().now().to_msg()

                camera_joints = []
                if track_ids is not None and results is not None:
                    # Publish skeleton data
                    camera_joints = self.publish_skeletons(track_ids, results)

                    # Publish ArUco transforms
                    if self.args.use_aruco:
                        self.publish_aruco_transforms(current_time)

                if self.args.display or self.args.render:
                    if self.args.render and camera_joints:
                        self._draw_overlay(frame, track_ids, results, camera_joints)
                    if track_ids is None:
                        cv2.putText(frame, 'No human detected', (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    cv2.imshow('front', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.get_logger().info('Quit requested by user')
                        break

                rclpy.spin_once(self, timeout_sec=0.001)

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user')
        except Exception as e:
            self.get_logger().error(f'Error in processing loop: {e}')
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info('Cleaning up resources...')

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()

        if hasattr(self, 'engine'):
            self.engine.close()

        self.marker_table.save()

        self.get_logger().info('Cleanup completed')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM3DBody-cpp ROS 2 node for real-time 3D human pose estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Engine configuration.  Defaults point at the SAM3DBody-cpp checkout this
    # package lives in, which keeps working through the workspace symlink.
    repo_root = repo_root_from(__file__)
    default_lib = os.path.join(repo_root, 'build') if repo_root else ''
    default_onnx = os.path.join(repo_root, 'onnx') if repo_root else ''

    parser.add_argument(
        '--lib-dir', type=str, default=default_lib,
        help='Directory containing libfast_sam_3dbody.so'
    )
    parser.add_argument(
        '--onnx-dir', type=str, default=default_onnx,
        help='Directory containing the ONNX/GGUF/LBS model files'
    )
    parser.add_argument(
        '--gguf', type=str, default='',
        help='Override path to pipeline.gguf (default: <onnx-dir>/pipeline.gguf)'
    )
    parser.add_argument(
        '--yolo', type=str, default='',
        help='Override path to yolo.onnx (default: <onnx-dir>/yolo.onnx)'
    )
    parser.add_argument(
        '--cuda', type=int, default=0,
        help='CUDA device index for the engine (-1 = CPU)'
    )
    parser.add_argument(
        '--refined-pose', action='store_true',
        help='Run the iterative refined-pose pass (per-hand decoder + wrist-IK '
             'splice), which gives markedly better hands at roughly half the '
             'frame rate. Needs the "refined" model profile: '
             'bash tools/fetch_model.sh cuda refined'
    )

    # Camera configuration
    parser.add_argument(
        '--input', type=str, default='/dev/video0',
        help='Input device or file (e.g. /dev/video0, /dev/video8, webcam, or a video file path)'
    )
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Camera capture width in pixels'
    )
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Camera capture height in pixels'
    )
    parser.add_argument(
        '--fps', type=int, default=15,
        help='Camera capture frame rate'
    )

    # Processing configuration
    parser.add_argument(
        '--detection-threshold', type=float, default=0.5,
        help='Confidence threshold for person detection (0.0-1.0)'
    )
    parser.add_argument(
        '--nms', type=float, default=0.45,
        help='NMS IoU threshold for person detection'
    )
    parser.add_argument(
        '--max-skeletons', type=int, default=0,
        help='Maximum number of people per frame (0 = unlimited)'
    )

    # Display and output options
    parser.add_argument(
        '--display', action='store_true',
        help='Show real-time video window (press "q" to quit)'
    )
    parser.add_argument(
        '--render', action='store_true',
        help='Draw the estimated skeleton, projected back into the image, on the '
             'OpenCV window (implies --display)'
    )
    parser.add_argument(
        '--output-folder', type=str, default='./logs',
        help='Output folder for the ArUco marker table'
    )

    # ArUco marker options
    parser.add_argument(
        '--use-aruco', action=argparse.BooleanOptionalAction, default=True,
        help='Enable ArUco marker detection for camera calibration. Pass '
             '--no-use-aruco to disable and publish in the raw camera frame'
    )
    parser.add_argument(
        '--aruco-marker-id', type=int, default=None,
        help='Only use this ArUco marker ID (DICT_6X6_250) as the skeleton '
             'reference, e.g. 1. Other markers are ignored. Default: any marker'
    )
    parser.add_argument(
        '--aruco-marker-length', type=float, default=0.15,
        help='Printed ArUco marker side length in meters (black border edge to edge). '
             'Scales the estimated camera translation'
    )
    parser.add_argument(
        '--aruco-parent-frame', type=str, default='wood_panel',
        help='TF parent frame the ArUco marker is mounted on'
    )
    parser.add_argument(
        '--aruco-xyz', type=float, nargs=3, default=[0.1055, 1.405, -0.1025],
        metavar=('X', 'Y', 'Z'),
        help='ArUco marker position in the parent frame, in meters'
    )
    parser.add_argument(
        '--aruco-rpy', type=float, nargs=3, default=[90.0, 0.0, 180.0],
        metavar=('ROLL', 'PITCH', 'YAW'),
        help='ArUco marker orientation in the parent frame, in degrees '
             '(fixed-axis roll about X, then pitch about Y, then yaw about Z). '
             'Marker axes: X right, Y up along the printed marker, Z out of the marker'
    )

    parser.add_argument(
        '--static-camera', action='store_true',
        help='The camera never moves, so the marker pose only changes when the robot '
             'does. Marker observations are then averaged per slider position, which '
             'removes detection noise, survives occlusion and lets detection be skipped '
             'once a position has converged'
    )
    parser.add_argument(
        '--static-robot', action='store_true',
        help='The robot never moves along its slider, so all observations belong to a '
             'single position and the slider topic is not needed (implies a slider of 0)'
    )
    parser.add_argument(
        '--slider-topic', type=str, default='/slider/position_y',
        help='std_msgs/Float64 topic carrying the robot slider position in meters'
    )
    parser.add_argument(
        '--slider-bin', type=float, default=0.01,
        help='Slider positions this far apart (meters) share one averaged marker pose'
    )
    parser.add_argument(
        '--marker-table-file', type=str, default='',
        help='Where the averaged marker poses are saved on shutdown and reloaded from '
             'on start. Default: <output-folder>/aruco_marker_table.json'
    )
    parser.add_argument(
        '--marker-min-samples', type=int, default=30,
        help='Observations a slider position needs before its averaged marker pose is '
             'trusted enough to skip detection'
    )
    parser.add_argument(
        '--marker-spread-threshold', type=float, default=0.01,
        help='Maximum spread (standard deviation, meters) of the averaged marker '
             'position for it to count as converged'
    )
    parser.add_argument(
        '--marker-recheck-interval', type=float, default=5.0,
        help='Once converged, how often (seconds) the marker is detected again to '
             'catch a bumped camera'
    )
    parser.add_argument(
        '--no-marker-line-fit', dest='marker_line_fit', action='store_false',
        help='Do not fit a straight line through the learned slider positions. By '
             'default the marker pose at a slider position that was never visited is '
             'interpolated from that line, which never overrides a position that has '
             'been measured directly'
    )
    parser.add_argument(
        '--marker-fit-residual', type=float, default=0.02,
        help='Largest residual (meters) the rail line fit may have before '
             'interpolation is refused, e.g. because the slider is not metric or the '
             'marker moved on its mount'
    )
    parser.add_argument(
        '--marker-alarm-pct', type=float, default=5.0,
        help='After loading poses from disk, warn while live observations disagree by '
             'more than this percentage of the marker distance (possible tampering '
             'while the node was down)'
    )

    # Camera intrinsics, at --width x --height.  Unlike D-PoSE these feed the
    # pose engine as well as the ArUco pose, so leaving them at 0 gives both the
    # value SAM-3D-Body was trained with: focal = image diagonal, centre = W/2, H/2.
    parser.add_argument('--fx', type=float, default=0.0, help='Camera focal length x in pixels (0 = image diagonal)')
    parser.add_argument('--fy', type=float, default=0.0, help='Camera focal length y in pixels (0 = image diagonal)')
    parser.add_argument('--cx', type=float, default=0.0, help='Camera principal point x in pixels (0 = width / 2)')
    parser.add_argument('--cy', type=float, default=0.0, help='Camera principal point y in pixels (0 = height / 2)')
    parser.add_argument(
        '--dist-coeffs', type=float, nargs='+', default=None,
        metavar='K',
        help='Lens distortion coefficients in OpenCV order (k1 k2 p1 p2 [k3 ...]). '
             'Default: no distortion'
    )

    # Camera retry option
    parser.add_argument(
        '--insist-camera', action='store_true',
        help='Retry camera initialization indefinitely (3 s delay between attempts) until it succeeds'
    )

    args = parser.parse_args()

    if args.render:
        args.display = True
    if not args.gguf:
        args.gguf = os.path.join(args.onnx_dir, 'pipeline.gguf')
    if not args.yolo:
        args.yolo = os.path.join(args.onnx_dir, 'yolo.onnx')
    return args


def validate_requirements(args):
    """Validate that all requirements are met."""
    errors = []

    if not args.lib_dir or not args.onnx_dir:
        errors.append(
            "Could not locate the SAM3DBody-cpp checkout from this script — "
            "pass --lib-dir and --onnx-dir explicitly.")

    required_files = [
        os.path.join(args.lib_dir, 'libfast_sam_3dbody.so'),
        args.gguf,
        args.yolo,
        os.path.join(args.onnx_dir, 'body_model.lbs'),
        os.path.join(args.onnx_dir, 'keypoint_mapping.bin'),
    ]
    if args.refined_pose:
        required_files.append(os.path.join(args.onnx_dir, 'pipeline_refined.gguf'))

    for file_path in required_files:
        if not os.path.exists(file_path):
            errors.append(f"Required file not found: {file_path}")

    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Please make sure you have:")
        print("   1. Run   bash setup.sh   in this package (builds the engine, fetches the models)")
        print("   2. Fetched the extra models --refined-pose needs, if you use it:")
        print("        bash tools/fetch_model.sh cuda refined")
        return False

    print("✅ All requirements validated successfully!")
    return True


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Validate requirements
    if not validate_requirements(args):
        return 1

    # Initialize ROS2
    rclpy.init()

    try:
        # Create and run the node
        node = PoseEstimationNode(args)
        node.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Shutdown ROS2
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
