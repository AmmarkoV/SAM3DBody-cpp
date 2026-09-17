import cv2
import numpy as np

# Step 1: Generate and save an ArUco marker
def generate_aruco_marker(marker_id=42, marker_size=200, file_name='marker_42.png'):
    # Define the dictionary to use (6x6 with 250 markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    # Save the marker image to a file
    cv2.imwrite(file_name, marker_image)
    print(f"Marker saved as {file_name}")



def detect_aruco_from_webcam():

    # Define marker real-world size in meters
    marker_length = 0.05  # 5 cm

    # Replace these with your actual calibration values
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0,   0,   1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

    # Load dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(4)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                # Estimate pose using solvePnP
                obj_points = np.array([
                    [-marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ], dtype=np.float32)

                img_points = corner[0].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                    print(f"Marker ID: {ids[i][0]}")
                    print("Rotation Vector (rvec):", rvec.flatten())
                    print("Translation Vector (tvec):", tvec.flatten())
                    print("-" * 30)

        cv2.imshow("ArUco Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def detect_aruco_from_image(frame, marker_id=None, fx=800.0, fy=800.0, cx=320.0, cy=240.0, dist_coeffs=None, marker_length=0.15, input_is_bgr=True):
    # marker_id: only use this marker for the pose (None = any marker, last detected wins)
    # marker_length: marker real-world side length in meters
    # input_is_bgr: set False when frame channels are already RGB (e.g. converted
    #   upstream for the pose model), otherwise the grayscale conversion below
    #   swaps the red/blue weighting and degrades corner localization, which
    #   biases the solvePnP pose (rvec/tvec) used for the marker<->camera transform

    rvec=None

    tvec=None

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
    else:
        dist_coeffs = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    # Load dictionary and detector

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if input_is_bgr else cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            if marker_id is not None and ids[i][0] != marker_id:
                continue

            # Estimate pose using solvePnP
            obj_points = np.array([
                [-marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0]
            ], dtype=np.float32)

            img_points = corner[0].astype(np.float32)
            success, marker_rvec, marker_tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

            if success:
                rvec, tvec = marker_rvec, marker_tvec
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                print(f"Marker ID: {ids[i][0]}")
                print("Rotation Vector (rvec):", rvec.flatten())
                print("Translation Vector (tvec):", tvec.flatten())
                print("-" * 30)

    return rvec,tvec

# Ported from D-PoSE's aruco/aruco_create.py (what magician_body_pose_estimation
# used) — unchanged except for this guard: upstream ran the three lines below at
# import time, which wrote marker PNGs into the working directory of every
# process that imported the module, the ROS node included.
if __name__ == '__main__':
    # Print these, stick one on the wall, and point --aruco-marker-id at its id.
    generate_aruco_marker(marker_id=1, marker_size=1200, file_name='marker_1.png')
    generate_aruco_marker(marker_id=13, marker_size=1200, file_name='marker_13.png')
    generate_aruco_marker(marker_id=42, marker_size=1200, file_name='marker_42.png')

    # Detect ArUco markers from webcam feed
    #detect_aruco_from_webcam()

