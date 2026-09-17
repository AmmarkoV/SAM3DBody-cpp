"""
Averaged ArUco marker poses per robot slider position.

With a static camera the marker -> camera transform only changes when the robot
moves along its 1D conveyor, so every observation taken at the same slider
position is a noisy measurement of the same pose.  This module accumulates
those observations into a running mean (per slider position) and hands back the
averaged pose, which stays valid while the marker is occluded and lets marker
detection be skipped once an entry has converged.

Ported verbatim from D-PoSE's aruco/marker_pose_table.py, the version
magician_body_pose_estimation used; it only needs numpy + OpenCV.
"""

import atexit
import json
import os
import signal
import time

import cv2
import numpy as np


def _rotmat_to_quat(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quat_to_rotmat(q):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_angle(q1, q2):
    """Angle in degrees between two quaternions, ignoring the q/-q sign."""
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


class _Entry:
    """Running mean of the marker pose observed at one slider position."""

    def __init__(self):
        self.n = 0.0
        self.sum_t = np.zeros(3)       # for the mean translation
        self.sum_t2 = np.zeros(3)      # for the per-axis spread
        self.M = np.zeros((4, 4))      # sum of q q^T, for the mean quaternion
        self.consecutive_rejects = 0
        self.last_detection = 0.0
        self.validation_left = 0       # >0 while a loaded entry is unproven
        self.validation_alarms = 0
        self.prior_t = None            # pose loaded from disk, kept for comparison

    def add(self, q, t, weight=1.0):
        if self.n > 0 and float(np.dot(q, self.mean_quat())) < 0:
            q = -q  # keep the accumulated quaternions on the same hemisphere
        self.n += weight
        self.sum_t += weight * t
        self.sum_t2 += weight * t * t
        self.M += weight * np.outer(q, q)

    def mean_t(self):
        return self.sum_t / self.n

    def mean_quat(self):
        # The mean rotation is the eigenvector of the largest eigenvalue of M.
        q = np.linalg.eigh(self.M)[1][:, -1]
        return q if q[0] >= 0 else -q

    def spread(self):
        """Standard deviation of the translation samples, in meters."""
        if self.n < 2:
            return float('inf')
        var = np.maximum(self.sum_t2 / self.n - self.mean_t() ** 2, 0.0)
        return float(np.sqrt(var.sum()))

    def uncertainty(self):
        """Standard error of the mean translation, in meters."""
        if self.n < 2:
            return float('inf')
        return self.spread() / np.sqrt(self.n)


class MarkerPoseTable:
    """
    Table of averaged marker poses, keyed by quantized slider position.

    Without static_camera this is a pass-through: every frame is detected and
    the latest observation is returned, which is the behaviour of the node when
    the camera may move.
    """

    def __init__(self, static_camera=False, static_robot=False, bin_size=0.01,
                 min_samples=30, spread_threshold=0.01, recheck_interval=5.0,
                 alarm_pct=5.0, prior_weight=20.0, table_file=None,
                 slider_timeout=5.0, line_fit=True, fit_residual=0.02,
                 logger=None):
        self.static_camera = static_camera
        self.static_robot = static_robot
        self.bin_size = bin_size
        self.min_samples = min_samples
        self.spread_threshold = spread_threshold
        self.recheck_interval = recheck_interval
        self.alarm_pct = alarm_pct
        self.prior_weight = prior_weight
        self.table_file = table_file
        self.slider_timeout = slider_timeout
        self.line_fit = line_fit
        self.fit_residual = fit_residual
        self.logger = logger

        self.entries = {}
        self.key = '0.000' if static_robot else None
        self.slider_value = 0.0
        self.slider_stamp = 0.0
        self._fit = None          # (t0, direction, mean_quat, residual)
        self._fit_dirty = True
        self._fit_reported = False
        self._warned_no_slider = False
        self._saved = False

        # Observations taken while the slider position is unknown cannot be
        # attributed to a position, so they feed a moving average instead.
        self.unknown_pose = None  # (quat, translation)

        if static_camera and table_file:
            self.load()

    # -- logging ----------------------------------------------------------
    def _log(self, level, msg):
        if self.logger is not None:
            getattr(self.logger, level)(msg)

    # -- slider -----------------------------------------------------------
    def set_slider(self, value):
        """Feed the latest slider position, in meters."""
        if self.static_robot:
            return
        self.slider_value = value
        key = f'{round(value / self.bin_size) * self.bin_size:.3f}'
        self.slider_stamp = time.time()
        if key != self.key:
            self._log('info', f'Slider at {value:.3f} m, marker pose entry {key} '
                              f'({"known" if key in self.entries else "new"})')
            self.key = key

    def _current_key(self):
        """Entry key for right now, or None while the slider is unknown."""
        if self.static_robot:
            return '0.000'
        if self.key is None or (time.time() - self.slider_stamp) > self.slider_timeout:
            if not self._warned_no_slider:
                self._log('warning',
                          'No slider position available: averaging the marker pose '
                          'with a moving average instead (the robot may be moving)')
                self._warned_no_slider = True
            return None
        return self.key

    # -- detection --------------------------------------------------------
    def should_detect(self):
        """Whether the marker still has to be detected in this frame."""
        if not self.static_camera:
            return True
        key = self._current_key()
        if key is None:
            return True
        entry = self.entries.get(key)
        if entry is None or not self.is_confident(entry):
            return True
        # Confident: re-check now and then, so a bumped camera is noticed.
        return (time.time() - entry.last_detection) >= self.recheck_interval

    def is_confident(self, entry):
        return (entry.n >= self.min_samples
                and entry.validation_left <= 0
                and entry.uncertainty() <= self.spread_threshold)

    def add(self, rvec, tvec):
        """Feed one marker observation from cv2.solvePnP."""
        t = np.asarray(tvec, dtype=float).reshape(3)
        q = _rotmat_to_quat(cv2.Rodrigues(np.asarray(rvec, dtype=float))[0])

        if not self.static_camera:
            self.unknown_pose = (q, t)
            return

        key = self._current_key()
        if key is None:
            # Slider unknown: exponential moving average, never trusted enough
            # to skip detection.
            if self.unknown_pose is None:
                self.unknown_pose = (q, t)
            else:
                pq, pt = self.unknown_pose
                if float(np.dot(q, pq)) < 0:
                    q = -q
                a = 0.1
                self.unknown_pose = (pq * (1 - a) + q * a, pt * (1 - a) + t * a)
            return

        entry = self.entries.setdefault(key, _Entry())
        entry.last_detection = time.time()
        self._fit_dirty = True

        if entry.n > 0:
            mean_t, mean_q = entry.mean_t(), entry.mean_quat()
            dist = float(np.linalg.norm(t - mean_t))
            angle = _quat_angle(q, mean_q)

            if entry.validation_left > 0:
                entry.validation_left -= 1
                # Compare against the pose from disk, not the running mean: the
                # mean already drifts towards the live observations.
                prior_dist = float(np.linalg.norm(t - entry.prior_t))
                tolerance = self.alarm_pct / 100.0 * float(np.linalg.norm(entry.prior_t))
                if prior_dist > tolerance:
                    entry.validation_alarms += 1
                    self._log('warning',
                              f'Marker at entry {key} is {prior_dist:.3f} m from the pose '
                              f'loaded from disk (over {self.alarm_pct:.1f}% of '
                              f'{np.linalg.norm(entry.prior_t):.3f} m): the setup may have '
                              'been moved since the last run, re-converging')
                if entry.validation_left == 0 and entry.validation_alarms >= self.min_samples / 2:
                    # The pose from disk was wrong far too often to be noise, so
                    # drop it and keep only what this run has seen.
                    self._log('warning',
                              f'Pose loaded from disk for entry {key} disagreed with '
                              f'{entry.validation_alarms} of {self.min_samples} live '
                              'observations: discarding it and relearning from scratch')
                    entry = _Entry()
                    entry.last_detection = time.time()
                    self.entries[key] = entry
            elif entry.n >= 5:
                # Reject outliers: solvePnP on a planar marker can flip, and a
                # partly occluded marker gives a badly placed pose.
                if dist > max(4.0 * entry.spread(), 0.03) or angle > 10.0:
                    entry.consecutive_rejects += 1
                    if entry.consecutive_rejects >= 20:
                        self._log('warning',
                                  f'Marker pose at entry {key} changed persistently '
                                  f'({dist:.3f} m, {angle:.1f} deg off): resetting the entry')
                        self.entries[key] = _Entry()
                        self.entries[key].last_detection = time.time()
                        self.entries[key].add(q, t)
                    return
            entry.consecutive_rejects = 0

        was_confident = self.is_confident(entry)
        entry.add(q, t)
        if not was_confident and self.is_confident(entry):
            self._log('info',
                      f'Marker pose at entry {key} converged after {int(entry.n)} '
                      f'observations (spread {entry.spread() * 1000:.1f} mm), '
                      f'detection now runs every {self.recheck_interval:.0f} s')

    # -- rail line fit ----------------------------------------------------
    def _fit_line(self):
        """
        Fit marker pose = t0 + slider * direction over all learned positions.

        Travel is 1D and the marker is rigid on the carriage, so its rotation
        in the camera is constant and its position moves along a straight line.
        Fitting that line lets observations at one slider position cover
        positions that were never visited.
        """
        if not self._fit_dirty:
            return self._fit
        self._fit_dirty = False
        self._fit = None

        points = [(float(key), entry) for key, entry in self.entries.items()
                  if entry.n >= 5 and entry.validation_left <= 0]
        if len(points) < 3:
            return None

        # Weighted least squares per axis, one point per learned position.
        w = np.array([entry.n for _, entry in points])
        x = np.array([slider for slider, _ in points])
        t = np.array([entry.mean_t() for _, entry in points])

        sw, sx, sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
        st = (w[:, None] * t).sum(axis=0)
        sxt = (w[:, None] * x[:, None] * t).sum(axis=0)
        den = sw * sxx - sx * sx
        if den <= 1e-9:
            return None  # all learned positions are effectively the same spot

        direction = (sw * sxt - sx * st) / den
        origin = (st - direction * sx) / sw
        residual = float(np.sqrt(
            (w * ((t - (origin + x[:, None] * direction)) ** 2).sum(axis=1)).sum() / sw))

        # Rotation is constant along the rail, so average every entry's
        # accumulated q q^T (the eigenvector method ignores the q/-q sign).
        M = sum(entry.M for _, entry in points)
        quat = np.linalg.eigh(M)[1][:, -1]
        quat = quat if quat[0] >= 0 else -quat

        if residual > self.fit_residual:
            if not self._fit_reported:
                self._log('warning',
                          f'Rail line fit residual is {residual * 1000:.0f} mm over '
                          f'{len(points)} positions, above the {self.fit_residual * 1000:.0f} mm '
                          'limit: not interpolating between slider positions')
                self._fit_reported = True
            return None

        if not self._fit_reported:
            self._log('info',
                      f'Rail line fit over {len(points)} slider positions: residual '
                      f'{residual * 1000:.1f} mm, {np.linalg.norm(direction):.3f} m of marker '
                      'travel per slider unit; unvisited positions are now interpolated')
            self._fit_reported = True

        self._fit = (origin, direction, quat, residual)
        return self._fit

    def estimate(self):
        """Best (rvec, tvec) for the current slider position, or None."""
        if not self.static_camera:
            return None if self.unknown_pose is None else self._as_rvec_tvec(*self.unknown_pose)
        key = self._current_key()
        entry = self.entries.get(key) if key is not None else None
        if entry is not None and self.is_confident(entry):
            return self._as_rvec_tvec(entry.mean_quat(), entry.mean_t())

        # This position is unmeasured or still converging: predict it from the
        # rail, which never overrides a position that was measured properly.
        if key is not None and self.line_fit:
            fit = self._fit_line()
            if fit is not None:
                origin, direction, quat, _ = fit
                slider = 0.0 if self.static_robot else self.slider_value
                return self._as_rvec_tvec(quat, origin + slider * direction)

        if entry is not None and entry.n > 0:
            return self._as_rvec_tvec(entry.mean_quat(), entry.mean_t())
        if self.unknown_pose is not None:
            return self._as_rvec_tvec(*self.unknown_pose)
        return None

    @staticmethod
    def _as_rvec_tvec(q, t):
        rvec = cv2.Rodrigues(_quat_to_rotmat(q))[0]
        return rvec, np.asarray(t, dtype=float).reshape(3, 1)

    # -- persistence ------------------------------------------------------
    def save(self):
        """Write the learned entries to disk (called on shutdown)."""
        if not self.table_file or not self.static_camera or self._saved:
            return
        self._saved = True
        data = {
            'bin_size': self.bin_size,
            'entries': {
                key: {
                    'n': entry.n,
                    'spread': entry.spread(),
                    't': entry.mean_t().tolist(),
                    'q': entry.mean_quat().tolist(),
                }
                for key, entry in self.entries.items() if entry.n > 0
            },
        }
        try:
            directory = os.path.dirname(self.table_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.table_file, 'w') as handle:
                json.dump(data, handle, indent=2)
            self._log('info', f'Saved {len(data["entries"])} marker pose entries '
                              f'to {self.table_file}')
        except OSError as e:
            self._log('warning', f'Could not save marker pose table: {e}')

    def load(self):
        """Load entries written by an earlier run, with reduced weight."""
        if not os.path.exists(self.table_file):
            return
        try:
            with open(self.table_file) as handle:
                data = json.load(handle)
        except (OSError, ValueError) as e:
            self._log('warning', f'Could not load marker pose table: {e}')
            return

        if abs(float(data.get('bin_size', self.bin_size)) - self.bin_size) > 1e-9:
            self._log('warning', 'Saved marker pose table uses a different slider bin '
                                 'size, ignoring it')
            return

        for key, saved in data.get('entries', {}).items():
            entry = _Entry()
            # Load with a capped weight so fresh observations can pull the mean
            # back if the setup was disturbed while the node was down.
            entry.add(np.array(saved['q'], dtype=float),
                      np.array(saved['t'], dtype=float),
                      weight=min(float(saved['n']), self.prior_weight))
            entry.validation_left = int(self.min_samples)
            entry.prior_t = entry.mean_t().copy()
            self.entries[key] = entry
        self._fit_dirty = True
        self._log('info', f'Loaded {len(self.entries)} marker pose entries from '
                          f'{self.table_file}; they are checked against live '
                          f'observations for the next {self.min_samples} detections')


def install_exit_hooks(table):
    """Save the table when the process exits or is terminated."""
    # atexit covers a normal exit and Ctrl+C (SIGINT stays with Python, so the
    # node keeps its KeyboardInterrupt cleanup path); these signals would
    # otherwise kill the process without saving.  SIGKILL cannot be caught.
    atexit.register(table.save)

    def _on_signal(signum, frame):
        table.save()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass  # not on the main thread, or signal unavailable
