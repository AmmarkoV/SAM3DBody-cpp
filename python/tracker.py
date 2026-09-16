import cv2
import numpy as np
import sys
import os
import glob
import csv
import argparse
from urllib.parse import parse_qs

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "200000")


def resize_to_fit_screen(img, max_w=1280, max_h=720, only_shrink=True):
    """
    Resize image to fit within (max_w, max_h) while preserving aspect ratio.
    - only_shrink=True: don't upscale small images.
    Returns (resized_img, scale).
    """
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h))

    if only_shrink and scale >= 1.0:
        return img, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), scale



# --------------------------------------------------
# Calibration (.calib) helpers (Stereolabs/Matlab format)
# --------------------------------------------------
def _next_data_line(lines, start_idx):
    """Return (idx, line) of next non-empty, non-comment line after start_idx."""
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if s != "" and not s.startswith("%"):
            return i, s
        i += 1
    return None, None


def load_stereolabs_calib(calib_path):
    """
    Parse the Stereolabs/Matlab-load compatible .calib format:
      %Width\n<val>\n%Height\n<val>\n%I\n(9 vals)\n%D\n(5 vals)
    Returns: (width, height, K(3x3 float32), dist(5x1 float32))
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        raw = f.readlines()

    width = height = None
    K_vals = []
    D_vals = []

    for i, line in enumerate(raw):
        s = line.strip()

        if s == "%Width":
            _, v = _next_data_line(raw, i + 1)
            if v is None:
                raise ValueError("Malformed calib file: missing Width value")
            width = int(float(v))

        elif s == "%Height":
            _, v = _next_data_line(raw, i + 1)
            if v is None:
                raise ValueError("Malformed calib file: missing Height value")
            height = int(float(v))

        elif s == "%I":
            K_vals = []
            j = i + 1
            while len(K_vals) < 9:
                j, v = _next_data_line(raw, j)
                if v is None:
                    raise ValueError("Malformed calib file: not enough intrinsics values under %I")
                K_vals.append(float(v))
                j += 1

        elif s == "%D":
            D_vals = []
            j = i + 1
            while len(D_vals) < 5:
                j, v = _next_data_line(raw, j)
                if v is None:
                    raise ValueError("Malformed calib file: not enough distortion values under %D")
                D_vals.append(float(v))
                j += 1

    if width is None or height is None:
        raise ValueError("Malformed calib file: Width/Height not found")
    if len(K_vals) != 9:
        raise ValueError("Malformed calib file: Intrinsics (%I) not found or incomplete")
    if len(D_vals) != 5:
        raise ValueError("Malformed calib file: Distortion (%D) not found or incomplete")

    K = np.array(K_vals, dtype=np.float32).reshape(3, 3)
    dist = np.array(D_vals, dtype=np.float32).reshape(5, 1)
    return width, height, K, dist


def scale_intrinsics(K, src_w, src_h, dst_w, dst_h):
    """
    Scale intrinsics from calibration resolution (src) to actual frame resolution (dst).
    This is a common practical fix when video is a scaled version of calibration resolution.
    """
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


# --------------------------------------------------
# ArUco marker generation
# --------------------------------------------------
def get_aruco_dictionary(dict_name: str):
    """
    dict_name example: 'DICT_6X6_250'
    """
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary '{dict_name}'. "
                         f"Expected something like DICT_6X6_250.")
    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def generate_aruco_marker(aruco_dict, marker_id=42, marker_size_px=200, file_path='marker_42.png'):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(marker_size_px))
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    cv2.imwrite(file_path, marker_image)
    print(f"[INFO] Marker saved: {file_path}")


def parse_id_list(s: str):
    """
    "1,13,42" -> [1, 13, 42]
    """
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return out


def parse_marker_lengths(s: str):
    """
    "1=0.06,3=0.12" -> {1:0.06, 3:0.12}
    """
    s = (s or "").strip()
    if not s:
        return {}

    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --marker-lengths token '{part}'. Use like 1=0.06,3=0.12")
        k, v = part.split("=", 1)
        out[int(k.strip())] = float(v.strip())
    return out


# --------------------------------------------------
# QR + math helpers
# --------------------------------------------------
qr = cv2.QRCodeDetector()


def rvec_to_euler_degrees(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def make_approx_camera_matrix(width, height):
    fx = fy = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    return K, dist

def make_chessboard_object_points(board_w, board_h, square_size_m):
    """
    board_w, board_h: number of INNER corners (columns, rows)
    square_size_m: physical size of one square (meters)
    Returns objp shape: (N, 3) float32
    """
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def detect_chessboard_pose(
    gray,
    frame_bgr,
    board_w,
    board_h,
    square_size_m,
    camera_matrix,
    dist_coeffs,
    use_sb=False,
    criteria=None
):
    """
    Returns: (found, corners, rvec, tvec)
      - corners: (N,1,2) float32 if found else None
      - rvec/tvec: (3,1) float64 if found+PnP ok else None
    Draws overlay on frame_bgr if found.
    """
    pattern_size = (int(board_w), int(board_h))

    if use_sb:
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    rvec = tvec = None

    if found and corners is not None:
        # Refine
        if criteria is not None:
            cv2.cornerSubPix(gray, corners, (20, 20), (-1, -1), criteria)

        # Draw corners
        cv2.drawChessboardCorners(frame_bgr, pattern_size, corners, found)

        # Pose
        objp = make_chessboard_object_points(board_w, board_h, square_size_m)

        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if ok:
            # draw axes using square size scale
            cv2.drawFrameAxes(frame_bgr, camera_matrix, dist_coeffs, rvec, tvec, float(square_size_m) * 3.0)

            # overlay text at first corner
            c0 = corners[0, 0].astype(int)
            x0, y0 = int(c0[0]), int(c0[1])

            rvec_flat = rvec.reshape(3).astype(np.float32)
            tvec_flat = tvec.reshape(3).astype(np.float32)
            roll, pitch, yaw = rvec_to_euler_degrees(rvec_flat)

            lines = [
                f"Chessboard {board_w}x{board_h} (sq={square_size_m*1000:.1f}mm)",
                f"T (m): x={tvec_flat[0]:+.3f} y={tvec_flat[1]:+.3f} z={tvec_flat[2]:+.3f}",
                f"RPY (deg): r={roll:+.1f} p={pitch:+.1f} y={yaw:+.1f}",
            ]

            ytxt = y0 - 40
            for line in lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame_bgr, (x0 - 3, ytxt - th - 3), (x0 + tw + 3, ytxt + 3), (0, 0, 0), -1)
                cv2.putText(frame_bgr, line, (x0, ytxt),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                ytxt += 18
        else:
            rvec = tvec = None

    return found, corners, rvec, tvec



def parse_video_source(arg):
    return int(arg) if arg.isdigit() else arg


def _parse_qr_kv_payload(txt):
    """
    Parse QR payloads like: t_perf_ms=...&t_unix_ms=...&frame=...&hz=...
    """
    if not txt:
        return {}
    qs = parse_qs(txt, keep_blank_values=True)
    out = {}
    for k, v in qs.items():
        out[k] = v[-1] if isinstance(v, list) and len(v) else ""
    return out

def qrcode_overlay(frame, gray, max_qr_side=1920):
    """
    Draw QR codes on frame and return (frame, decoded_texts:list[str]).
    Robust to OpenCV QRCodeDetector occasional crashes.
    Uses optional downscale for stability/speed.
    """
    decoded_texts = []

    if gray is None or gray.size == 0:
        return frame, decoded_texts

    # Ensure gray is uint8
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)

    h, w = gray.shape[:2]

    # Downscale for QR detection stability (but not too aggressive)
    scale = 1.0
    gray_qr = gray
    if max(h, w) > max_qr_side:
        scale = float(max_qr_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray_qr = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Make contiguous (helps after remap/resize)
    gray_qr = np.ascontiguousarray(gray_qr)

    def _draw_poly(pts, label):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if scale != 1.0:
            pts /= scale
        pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)

        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        x0, y0 = pts[0]
        cv2.putText(frame, label, (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Try Multi decode, but NEVER early-return on failure ---
    multi_failed = False
    if hasattr(qr, "detectAndDecodeMulti"):
        try:
            try:
                ok, decoded_info, points, _ = qr.detectAndDecodeMulti(gray_qr)
            except ValueError:
                ok, decoded_info, points = qr.detectAndDecodeMulti(gray_qr)

            if points is not None and decoded_info is not None:
                for txt, pts in zip(decoded_info, points):
                    if txt is None or txt.strip() == "":
                        continue
                    decoded_texts.append(txt)
                    _draw_poly(pts, f"QR: {txt}")

        except cv2.error:
            multi_failed = True

    # --- Fallback single decode if multi found nothing OR multi errored ---
    if len(decoded_texts) == 0:
        try:
            txt, pts, _ = qr.detectAndDecode(gray_qr)
            if pts is not None and txt and txt.strip() != "":
                decoded_texts.append(txt)
                _draw_poly(pts, f"QR: {txt}")
        except cv2.error:
            # Skip QR for this frame
            pass

    return frame, decoded_texts



# --------------------------------------------------
# CSV writer (wide format)
# --------------------------------------------------
def _fmt_float(x, ndigits):
    try:
        if x == '' or x is None:
            return ''
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return ''


def write_wide_csv(frame_records, csv_path):
    """
    Write a *wide* CSV: columns sized to maximum number of ArUco markers / QRs seen in any frame.

    frame_records: list of dicts like:
      {
        'frame_id': int,
        'aruco': [ {'id':int,'tvec':(x,y,z),'rpy':(r,p,y)} , ... ],
        'qr':    [ {'raw':str,'t_perf_ms':str,'t_unix_ms':str,'frame':str,'hz':str}, ...]
      }
    """
    max_aruco = max((len(r.get('aruco', [])) for r in frame_records), default=0)
    max_qr = max((len(r.get('qr', [])) for r in frame_records), default=0)

    header = ['frame_id']

    for i in range(max_aruco):
        header += [
            f'aruco_{i}_id',
            f'aruco_{i}_tvec_x_m', f'aruco_{i}_tvec_y_m', f'aruco_{i}_tvec_z_m',
            f'aruco_{i}_roll_deg', f'aruco_{i}_pitch_deg', f'aruco_{i}_yaw_deg',
        ]

    for i in range(max_qr):
        header += [
            f'qr_{i}_raw',
            f'qr_{i}_t_perf_ms',
            f'qr_{i}_t_unix_ms',
            f'qr_{i}_frame',
            f'qr_{i}_hz',
        ]

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)

        for r in frame_records:
            row = [r.get('frame_id', '')]

            ar = r.get('aruco', [])
            for i in range(max_aruco):
                if i < len(ar):
                    m = ar[i]
                    tid = m.get('id', '')
                    tvec = m.get('tvec', ('', '', ''))
                    rpy = m.get('rpy', ('', '', ''))
                    row += [
                        tid,
                        _fmt_float(tvec[0], 6), _fmt_float(tvec[1], 6), _fmt_float(tvec[2], 6),
                        _fmt_float(rpy[0], 3), _fmt_float(rpy[1], 3), _fmt_float(rpy[2], 3),
                    ]
                else:
                    row += [''] * 7

            qrs = r.get('qr', [])
            for i in range(max_qr):
                if i < len(qrs):
                    q = qrs[i]
                    row += [
                        q.get('raw', ''),
                        q.get('t_perf_ms', ''),
                        q.get('t_unix_ms', ''),
                        q.get('frame', ''),
                        q.get('hz', ''),
                    ]
                else:
                    row += [''] * 5

            w.writerow(row)


# --------------------------------------------------
# ArUco detection + pose + optional rectification + optional frame dumping
# --------------------------------------------------
def detect_aruco(
    video_source,
    aruco_dict_name="DICT_6X6_250",
    marker_lengths_m=None,
    default_marker_length_m=0.05,
    calib_file="",
    rectify_alpha=1.0,
    save_video=True,
    output_video_name="livelastRun3DHiRes.mp4",
    frame_prefix="colorFrame",
    image_ext="jpg",
    max_consecutive_failures=30,
    # --- chessboard ---
    track_chessboard=False,
    cb_w=9,
    cb_h=6,
    cb_square=0.024,
    cb_use_sb=False,
):
    """
    marker_lengths_m: dict[int -> float] mapping marker id -> physical side length in meters.
    If calib_file is provided, frames are undistorted/rectified before detection and pose uses newK + zero distortion.
    """
    if marker_lengths_m is None:
        marker_lengths_m = {}

    # Base name used for outputs (mp4 archival copy + csv)
    video_source_str = str(video_source)
    base_no_ext = os.path.splitext(video_source_str)[0]
    csv_path = f"{base_no_ext}.csv"

    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    cb_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)


    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30.0

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    h, w = frame.shape[:2]

    # Defaults: approximate pinhole, no distortion
    camera_matrix, dist_coeffs = make_approx_camera_matrix(w, h)

    # Optional rectification maps
    use_rectify = False
    mapx = mapy = None

    if calib_file and str(calib_file).strip() != "":
        try:
            cal_w, cal_h, K_cal, D_cal = load_stereolabs_calib(calib_file)

            # Scale intrinsics if needed
            if (cal_w != w) or (cal_h != h):
                print(f"[INFO] calib {cal_w}x{cal_h} != video {w}x{h} -> scaling intrinsics")
                K_use = scale_intrinsics(K_cal, cal_w, cal_h, w, h)
            else:
                K_use = K_cal

            newK, _ = cv2.getOptimalNewCameraMatrix(K_use, D_cal, (w, h), alpha=float(rectify_alpha))
            mapx, mapy = cv2.initUndistortRectifyMap(
                K_use, D_cal, None, newK, (w, h), m1type=cv2.CV_32FC1
            )

            # Rectify first frame
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Pose estimation on rectified image: use newK, no distortion
            camera_matrix = newK.astype(np.float32)
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)

            use_rectify = True
            print(f"[INFO] Using calib file: {calib_file} (rectification ENABLED)")
        except Exception as e:
            print(f"[WARN] Failed to load/apply calib_file='{calib_file}': {e}")
            print("[WARN] Falling back to approximate camera (rectification DISABLED)")

    frame_id = 1
    consecutive_failures = 0
    frame_records = []

    try:
        while True:
            # Process current (known-good) frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            frame, qr_texts = qrcode_overlay(frame, gray)

            frame_rec = {'frame_id': frame_id, 'aruco': [], 'qr': []}

            # --- Chessboard tracking (optional) ---
            if track_chessboard:
                found_cb, corners_cb, rvec_cb, tvec_cb = detect_chessboard_pose(
                    gray=gray,
                    frame_bgr=frame,
                    board_w=cb_w,
                    board_h=cb_h,
                    square_size_m=cb_square,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    use_sb=cb_use_sb,
                    criteria=cb_criteria
                )

                # Optional: store in frame record (similar to ArUco)
                if found_cb and (rvec_cb is not None) and (tvec_cb is not None):
                    rvec_flat = rvec_cb.reshape(3).astype(np.float32)
                    tvec_flat = tvec_cb.reshape(3).astype(np.float32)
                    roll, pitch, yaw = rvec_to_euler_degrees(rvec_flat)
                    frame_rec.setdefault('chessboard', [])
                    frame_rec['chessboard'].append({
                        'w': int(cb_w),
                        'h': int(cb_h),
                        'square_m': float(cb_square),
                        'tvec': (float(tvec_flat[0]), float(tvec_flat[1]), float(tvec_flat[2])),
                        'rpy': (float(roll), float(pitch), float(yaw)),
                    })


            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    marker_id_int = int(marker_id)
                    L = float(marker_lengths_m.get(marker_id_int, default_marker_length_m))

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], L, camera_matrix, dist_coeffs
                    )

                    rvec = rvecs[0].reshape(3)
                    tvec = tvecs[0].reshape(3)

                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, L * 0.5)

                    roll, pitch, yaw = rvec_to_euler_degrees(rvec)
                    frame_rec['aruco'].append({
                        'id': marker_id_int,
                        'tvec': (float(tvec[0]), float(tvec[1]), float(tvec[2])),
                        'rpy': (float(roll), float(pitch), float(yaw)),
                    })

                    # On-screen overlay near first corner
                    c = corners[i].reshape(4, 2).astype(int)
                    x0, y0 = c[0]

                    lines = [
                        f"ID {marker_id_int} (L={L*100:.1f}cm)",
                        f"T (m): x={tvec[0]:+.3f} y={tvec[1]:+.3f} z={tvec[2]:+.3f}",
                        f"RPY (deg): r={roll:+.1f} p={pitch:+.1f} y={yaw:+.1f}"
                    ]

                    ytxt = y0 - 30
                    for line in lines:
                        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(frame, (x0 - 3, ytxt - th - 3), (x0 + tw + 3, ytxt + 3), (0, 0, 0), -1)
                        cv2.putText(frame, line, (x0, ytxt),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        ytxt += 18

            # Parse QR payloads
            for txt in qr_texts:
                d = _parse_qr_kv_payload(txt)
                frame_rec['qr'].append({
                    'raw': txt,
                    't_perf_ms': d.get('t_perf_ms', ''),
                    't_unix_ms': d.get('t_unix_ms', ''),
                    'frame': d.get('frame', ''),
                    'hz': d.get('hz', ''),
                })

            frame_rec['aruco'].sort(key=lambda x: x.get('id', 0))
            frame_records.append(frame_rec)

            if save_video:
                fname = f"{frame_prefix}_{frame_id:05d}.{image_ext}"
                cv2.imwrite(fname, frame)

            frameSmaller,scale = resize_to_fit_screen(frame)
            cv2.imshow("FORTH All in one marker tracker", frameSmaller)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Read next frame (tolerate corrupted frames)
            ret, next_frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    print(f"[WARN] Stopping: exceeded max_consecutive_failures={max_consecutive_failures} "
                          f"(last streak={consecutive_failures}).")
                    break
                continue

            consecutive_failures = 0

            if use_rectify:
                next_frame = cv2.remap(next_frame, mapx, mapy, cv2.INTER_LINEAR)

            frame = next_frame
            frame_id += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Write CSV
    write_wide_csv(frame_records, csv_path)
    print(f"[INFO] Wrote CSV: {csv_path}")

    # Encode video from dumped frames (ffmpeg)
    if save_video:
        ffmpeg_cmd = (
            f"ffmpeg -framerate {int(fps)} "
            f"-start_number 1 "
            f"-i {frame_prefix}_%05d.{image_ext} "
            f"-s {w}x{h} "
            f"-y -r {int(fps)} "
            f"-pix_fmt yuv420p "
            f"-threads 8 "
            f"{output_video_name}"
        )
        print("\n[INFO] Running ffmpeg:")
        print(ffmpeg_cmd)
        os.system(ffmpeg_cmd)

        out_mp4 = f"{base_no_ext}_out.mp4"
        os.system(f"cp {output_video_name} {out_mp4}")
        print(f"[INFO] Copied: {output_video_name} -> {out_mp4}")

        for f in glob.glob(f"{frame_prefix}_*.{image_ext}"):
            os.remove(f)


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ArUco + QR tracking with optional rectification via .calib")

    # Positional video source (like your original script)
    p.add_argument("video_source", help="Video path or camera index (e.g. 0)")

    # Optional calibration/rectification
    p.add_argument("--calib", default="", help="Optional .calib file (Stereolabs/Matlab format) for rectification")
    p.add_argument("--rectify-alpha", type=float, default=1.0,
                   help="Rectify alpha: 1.0 keeps more FOV (black borders), 0.0 crops more")

    # Output / robustness
    p.add_argument("--output-video", default="livelastRun3DHiRes.mp4", help="Output video filename (mp4)")
    p.add_argument("--max-failures", type=int, default=30, help="Max consecutive failed frame reads before stopping")
    p.add_argument("--save-video", action="store_true", default=True, help="Save annotated video via frame dump+ffmpeg")
    p.add_argument("--no-save-video", action="store_false", dest="save_video", help="Disable saving video")

    p.add_argument("--frame-prefix", default="colorFrame", help="Prefix for dumped frames")
    p.add_argument("--image-ext", default="jpg", help="Frame image extension (jpg/png)")

    # ArUco configuration
    p.add_argument("--aruco-dict", default="DICT_6X6_250", help="OpenCV ArUco dictionary name")
    p.add_argument("--marker-lengths", default="",
                   help="Marker side lengths in meters per id, e.g. '1=0.06,3=0.12'")
    p.add_argument("--default-marker-length", type=float, default=0.05,
                   help="Default marker side length in meters if id not specified")

    # Marker generation
    p.add_argument("--generate-markers", default="",
                   help="Comma-separated marker IDs to generate PNGs for, e.g. '1,13,42'")
    p.add_argument("--marker-img-size", type=int, default=200, help="Generated marker image size in pixels")
    p.add_argument("--marker-out-dir", default=".", help="Directory to write generated marker PNGs")

    # Chessboard tracking (optional)
    p.add_argument("--track-chessboard", action="store_true", default=False,
                   help="Enable chessboard detection + pose tracking")
    p.add_argument("--cb-w", type=int, default=9,
                   help="Chessboard INNER corners width (columns). Example: 9")
    p.add_argument("--cb-h", type=int, default=6,
                   help="Chessboard INNER corners height (rows). Example: 6")
    p.add_argument("--cb-square", type=float, default=0.024,
                   help="Chessboard square size in meters. Example: 0.024 for 24mm")
    p.add_argument("--cb-sb", action="store_true", default=False,
                   help="Use findChessboardCornersSB (more robust, often slower)")



    return p.parse_args()


def main():
    args = parse_args()

    video_source = parse_video_source(args.video_source)
    marker_lengths_m = parse_marker_lengths(args.marker_lengths)

    # Generate marker images if requested
    if args.generate_markers.strip():
        ids = parse_id_list(args.generate_markers)
        aruco_dict = get_aruco_dictionary(args.aruco_dict)
        for mid in ids:
            out_path = os.path.join(args.marker_out_dir, f"marker_{mid}.png")
            generate_aruco_marker(aruco_dict, marker_id=mid, marker_size_px=args.marker_img_size, file_path=out_path)

    detect_aruco(
        video_source=video_source,
        aruco_dict_name=args.aruco_dict,
        marker_lengths_m=marker_lengths_m,
        default_marker_length_m=args.default_marker_length,
        calib_file=args.calib,
        rectify_alpha=args.rectify_alpha,
        save_video=args.save_video,
        output_video_name=args.output_video,
        frame_prefix=args.frame_prefix,
        image_ext=args.image_ext,
        max_consecutive_failures=args.max_failures,
        # --- chessboard ---
        track_chessboard=args.track_chessboard,
        cb_w=args.cb_w,
        cb_h=args.cb_h,
        cb_square=args.cb_square,
        cb_use_sb=args.cb_sb,
    )


if __name__ == "__main__":
    main()

