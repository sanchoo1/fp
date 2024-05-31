import cv2
import cv2.aruco as aruco
import C2D
import numpy as np

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

import util


def invert_transform(R, t):
    """ Invert a transformation given a rotation matrix R and translation t. """
    R, _ = cv2.Rodrigues(R)
    R_inv = np.transpose(R)
    t_inv = -np.dot(R_inv, t)
    return R_inv, t_inv

def compose_transformations(R_A, t_A, R_B, t_B):
    # Convert rotation vectors to rotation matrices if necessary
    R_A, _ = cv2.Rodrigues(R_A)
    R_B, _ = cv2.Rodrigues(R_B)

    # Calculate the composed rotation matrix
    R_C = np.dot(R_B, R_A)
    
    # Calculate the composed translation vector
    t_C = np.dot(R_B, t_A) + t_B
    
    return R_C, t_C

def get_marker_world_positions(marker_size, layout_size):
    """Calculate world positions of markers based on layout size and marker size."""
    half_layout = layout_size / 2
    half_marker = marker_size / 2
    return {
        101: np.array([-half_layout + half_marker,  half_layout - half_marker, 0]), # top left
        102: np.array([ half_layout - half_marker,  half_layout - half_marker, 0]), # top right
        103: np.array([-half_layout - half_marker, -half_layout + half_marker, 0]), # lower left
        104: np.array([ half_layout + half_marker, -half_layout + half_marker, 0]), # lower right
    }

def pnp_get_markers_pose(dictionary, parameters, frame, camera_matrix, dist_coeffs, marker_length):

    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)

    # Define 3D points for markers in the world coordinate system (assuming markers are on the xy-plane)
    marker_points = {
        101: np.array([[-0.5, -0.5, 0], [-0.5 + marker_length, -0.5, 0], 
                     [-0.5 + marker_length, -0.5 + marker_length, 0], [-0.5, -0.5 + marker_length, 0]], dtype=np.float32),
        102: np.array([[0.5 - marker_length, -0.5, 0], [0.5, -0.5, 0], 
                     [0.5, -0.5 + marker_length, 0], [0.5 - marker_length, -0.5 + marker_length, 0]], dtype=np.float32),
        103: np.array([[0.5 - marker_length, 0.5 - marker_length, 0], [0.5, 0.5 - marker_length, 0], 
                     [0.5, 0.5, 0], [0.5 - marker_length, 0.5, 0]], dtype=np.float32),
        104: np.array([[-0.5, 0.5 - marker_length, 0], [-0.5 + marker_length, 0.5 - marker_length, 0], 
                     [-0.5 + marker_length, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
    }

    # If markers detected
    if ids is not None:
        all_rvecs, all_tvecs = [], []
        for i in range(ids.size):
            marker_id = ids[i, 0]
            if marker_id in marker_points:
                object_points = marker_points[marker_id]
                image_points = corners[i][0]

                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
                if success:
                    all_rvecs.append(rvec)
                    all_tvecs.append(tvec)
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length / 2)

        if all_rvecs:
            
            # Optionally, average the rvecs and tvecs if needed
            mean_rvec = np.mean(all_rvecs, axis=0)
            mean_tvec = np.mean(all_tvecs, axis=0)
            print("Average Rotation Vector:", mean_rvec.ravel())
            print("Average Translation Vector:", mean_tvec.ravel())

    return frame


def get_camera_pose_from_aruco_markers(dictionary, parameters, frame, camera_matrix, dist_coeffs, detector, marker_length):

    
    # Predefined positions of markers in world coordinates (modify these based on your setup)
    marker_positions = get_marker_world_positions(marker_length, 0.25)
    # Detect Aruco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    aruco.drawDetectedMarkers(frame, corners, ids)
    all_transforms = []
    temp1 = frame.copy()

    if len(corners) > 0:
        # Estimate pose for each detected marker
        for corner, marker_id in zip(corners, ids.flatten()):
            if marker_id in marker_positions:
                # Estimate pose of the marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)
                rvec = rvec[0][0]  # Simplify the rvec array shape
                tvec = tvec[0][0]  # Simplify the tvec array shape

                # Transformation from camera to marker
                R_cm, _ = cv2.Rodrigues(rvec)
                t_cm = tvec.reshape((3, 1))

                # Marker to camera transformation matrix
                T_mc = np.vstack((np.hstack((R_cm, t_cm)), [0, 0, 0, 1]))

                R_mc = T_mc[:3, :3]
                t_mc = T_mc[:3, 3].reshape((3, 1))
                R_mc, _ = cv2.Rodrigues(R_mc)

                # Draw the axes at the workspace origin
                cv2.drawFrameAxes(temp1, camera_matrix, dist_coeffs, R_mc, t_mc, 0.01)

                # World to marker transformation matrix
                t_wm = np.array(marker_positions[marker_id]).reshape((3, 1))
                R_wm = np.eye(3)  # Assuming no rotation of the markers in the world
                T_mw = np.vstack((np.hstack((R_wm, t_wm)), [0, 0, 0, 1]))

                # Compute the transformation from world to camera
                T_wc = np.dot(T_mc, np.linalg.inv(T_mw))  # World to camera transformation matrix
                all_transforms.append(T_wc)

        # Optional: Compute average transformation from world origin to camera
        if all_transforms:
            avg_T_wc = np.mean(np.stack(all_transforms), axis=0)

            # Extract rotation vector and translation vector from average transformation matrix
            R_wc = avg_T_wc[:3, :3]
            t_wc = avg_T_wc[:3, 3].reshape((3, 1))
            rvec_wc, _ = cv2.Rodrigues(R_wc)

            # Draw the axes at the workspace origin
            temp1 = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_wc, t_wc, 0.05)
            cv2.imshow('2', temp1)
            return avg_T_wc  # c = Twc * w,  c = RT_dc * d
        else:
            return None
    else:
        return None

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)


param = C2D.Params()
cameraMatrix, distCoeffs = param.c_intrinsics, param.c_distor

counter = 0


k = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)



cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('1', 1920, 1080)
cv2.namedWindow('2')

while True:
    c_frame = k.get_last_color_frame()
    c_frame = np.reshape(c_frame, [1080, 1920, 4])[:, :, 0:3]
    c_frame = cv2.flip(c_frame, 1)
    cv2.imshow('1', c_frame)
    
    aru = util.MyAruco()
    # frame = pnp_get_markers_pose(dictionary, parameters, frame, cameraMatrix, distCoeffs, 0.032)
    # t = aru.get_camera_pose_from_aruco_markers(c_frame)
    t = get_camera_pose_from_aruco_markers(dictionary, parameters, c_frame, param.c_intrinsics, param.c_distor, detector, 0.025)

    # if counter % 100 == 0:
    #     print (t[0])

    # counter += 1 

    # display
    



    if cv2.waitKey(1) == ord('q'):
        break






# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

#     # 如果检测到至少一个Aruco标记
#     if len(corners) > 0:
#         # 估计标记的姿态
#         rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
        
#         # 可视化姿态
#         for i in range(len(rvec)):
#             cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.03)
#             aruco.drawDetectedMarkers(frame, corners, ids)

#     # 显示图像
#     cv2.imshow('1', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# 释放摄像头资源
# cap.release()
# cv2.destroyAllWindows()