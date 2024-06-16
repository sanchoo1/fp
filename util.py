import cv2
import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime
import cv2.aruco as aruco
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

class Yolo8:
    def __init__(self):
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        self.model = YOLO( ROOT / 'yoloviii/xxx.pt')     
    
    def annotate(self, frame):
        cropped = frame[12:1068, 432:1488]
        results = self.model(cropped)
        annotated_frame = results[0].plot()
        return annotated_frame


class MyKinect:
    def __init__(self):

        """ self.d_intrinsics_mat = np.array([[362.4679, 0, 261.4258], [0, 361.1774, 208.1123], [0, 0, 1]])
        self.c_intrinsics_mat = np.array([[1038.6, 0, 951.1293], [0, 1038.9, 526.1001], [0, 0, 1]])
        self.inv_d_intrinsics_mat = np.array([[0.00275886, 0, -0.72123849],
                                          [0, 0.00276872, -0.57620521],
                                          [0, 0, 1]])
        self.inv_c_intrinsics_mat = np.array([[0.00096283, 0, -0.91578019],
                                          [0, 0.00096256, -0.50640110],
                                          [0, 0, 1]]) """
        self.d_intrinsics = np.array([[367.816, 0, 256], [0, 367.816, 212], [0, 0, 1]])
        self.c_intrinsics = np.array([[1066.6667, 0, 960], [0, 1066.667, 540], [0, 0, 1]])
        self.inv_d_intrinsics = np.array([[0.002718, 0, -0.696224], [0, 0.002718, -0.576228], [0, 0, 1]])
        self.inv_c_intrinsics = np.array([[0.0009375, 0, -0.89999997],
                                          [0, 0.0009375, -0.50624984],
                                          [0, 0, 1]])
        """ self.RT_d_to_c = np.array([[0.999820242867276, -0.005590505683472, 0.018117069272502, -51.337633677812520],
                           [0.005420373979096, 0.999940881177058, 0.009426223887326, 3.371261769890467],
                           [-0.018168695570907, -0.009326328165486, 0.999791437302901, 17.491786028014985],
                           [0, 0, 0,1]]) """       

        self.RT_d_to_c_reg = np.array([
                           [1, 0, 0, -51.34],[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.d_distor = np.array([0.1174, 0.2216, 0, 0, 0])
        self.c_distor = np.array([0.0378, 0.1169, 0, 0, 0])

        # logi
        self.intrinsics = np.array([[820.177613, -0.085439, 329.445492], [0, 819.003601, 243.131495], [0, 0, 1]])
        self.distor = np.array([0.002196, 0.703961, 0.000977, 0.000463, -1.335740])


    def get_color_frame(self):
        c_frame = self.kinect.get_last_color_frame()
        c_frame = np.reshape(c_frame, (1080, 1920, 4))[:, :, 0:3]
        c_frame = cv2.flip(c_frame, 1)
        return c_frame

    def get_depth_frame(self):
        d_frame = self.kinect.get_last_depth_frame()
        d_frame = np.reshape(d_frame, (424, 512))
        d_frame = cv2.flip(d_frame, 1)
        return d_frame
    
    def generateD(self, d_frame):
        temp_frame = np.full((((1080, 1920, 3))), 0)

        for row in range(424):
            for col in range(512):
                if (d_frame[row][col] <= 5000 and d_frame[row][col]>0):
                    u_f, v_f = self.d_to_c(row, col, d_frame[row][col])
                    x = int(u_f)
                    y = int(v_f)

                    if not (x < 0 or x >= 1920 or y >= 1080 or y < 0):
                        temp_frame[y, x] = d_frame[row][col], col, row

        # cv2.imshow("cd", drawDepth(temp_frame[:, :, 0]))
        # cv2.waitKey(10)

        return temp_frame, drawDepth(temp_frame[:, :, 0])
    
    # def d_2_c(self, y, x, z):
    #     yw = (y - 212) / 367.816 * z
    #     xw = (x - 256) / 367.816 * z
    #     d_world_coordinate = np.array([[xw], [yw], [z], [1]])
    #     c_world_coordinate = np.dot(self.RT_d_to_c_reg, d_world_coordinate)
    #     c_pixel_y = 1066.666666667 / z * c_world_coordinate[1,0] + 540
    #     c_pixel_x = 1066.666666667 / z * c_world_coordinate[0,0] + 960
    #     return c_pixel_x, c_pixel_y
    
    def d_to_c(self, y, x, z):
        d_camera_coordinate = np.vstack((self.inv_d_intrinsics @ np.array([[x], [y], [1]]) * z, [1]))
        c_camera_coordinate = np.dot(self.RT_d_to_c_reg, d_camera_coordinate)
        c_camera_coordinate = c_camera_coordinate[:3]
        c_pixel = self.c_intrinsics @ c_camera_coordinate / z
        return c_pixel[0], c_pixel[1]

    
    def getPosition(self, d_camera_coordinate, RT):
        # print(d_camera_coordinate)

        c_cam = self.RT_d_to_c_reg @ d_camera_coordinate
        world_Pos = np.linalg.inv(RT) @ c_cam
        return world_Pos
    
    def getCameraCoo(self, array):
        z, x, y = array.tolist()
        d_camera_coordinate = np.vstack((self.inv_d_intrinsics @ np.array([[x], [y], [1]]) * z, [1]))
        return d_camera_coordinate

class MyAruco:
    def __init__(self, param):
        self.cameraMatrix, self.distCoeffs = param.c_intrinsics, param.c_distor

        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    def get_marker_world_positions(self, marker_size, length, width):
        """Calculate world positions of markers based on layout size and marker size."""

        half_marker = marker_size / 2
        return {
            101: np.array([-length / 2 + half_marker,  width / 2 - half_marker, 0]), # top left
            102: np.array([ length / 2 - half_marker,  width / 2 - half_marker, 0]), # top right
            103: np.array([-length / 2 + half_marker, -width / 2 + half_marker, 0]), # lower left
            104: np.array([ length / 2 - half_marker, -width / 2 + half_marker, 0]), # lower right
        }
    
    def detected(self, frame):
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, corners, ids)
        return corners, ids, frame

    def get_camera_pose_from_aruco_markers(self, frame, marker_size=60):

        # Predefined positions of markers in world coordinates (modify these based on your setup)
        marker_positions = self.get_marker_world_positions(marker_size, 450, 300)
        # Detect Aruco markers in the frame
        corners, ids, frame = self.detected(frame)
        
        all_transforms = []
        temp1 = frame.copy()

        if len(corners) > 0:
            # Estimate pose for each detected marker
            for corner, marker_id in zip(corners, ids.flatten()):
                if marker_id in marker_positions:
                    # Estimate pose of the marker
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_size, self.cameraMatrix, self.distCoeffs)
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
                    # cv2.drawFrameAxes(temp1, self.cameraMatrix, self.distCoeffs, R_mc, t_mc, 0.01)

                    # World to marker transformation matrix
                    t_wm = np.array(marker_positions[marker_id]).reshape((3, 1))
                    R_wm = np.eye(3)  # Assuming no rotation of the markers in the world
                    T_mw = np.vstack((np.hstack((R_wm, t_wm)), [0, 0, 0, 1]))

                    # Compute the transformation from world to camera
                    T_wc = np.dot(T_mc, np.linalg.inv(T_mw))  # World to camera transformation matrix
                    all_transforms.append(T_wc)

            if all_transforms:
                # 将所有旋转矩阵转换为四元数
                quaternions = [R.from_matrix(T[:3, :3]).as_quat() for T in all_transforms]

                # 平均四元数
                avg_quat = np.mean(quaternions, axis=0)
                avg_quat /= np.linalg.norm(avg_quat)  # 归一化

                # 将平均四元数转换回旋转矩阵
                avg_R_wc = R.from_quat(avg_quat).as_matrix()

                # 平均平移向量
                avg_t_wc = np.mean([T[:3, 3] for T in all_transforms], axis=0)

                # 重新构建平均变换矩阵
                avg_T_wc = np.eye(4)
                avg_T_wc[:3, :3] = avg_R_wc
                avg_T_wc[:3, 3] = avg_t_wc

                # 提取旋转向量和平移向量
                rvec_wc, _ = cv2.Rodrigues(avg_R_wc)
                t_wc = avg_t_wc.reshape((3, 1))
                
                # avg_T_wc = np.mean(np.stack(all_transforms), axis=0)

                # # Extract rotation vector and translation vector from average transformation matrix
                # R_wc = avg_T_wc[:3, :3]
                # t_wc = avg_T_wc[:3, 3].reshape((3, 1))
                # rvec_wc, _ = cv2.Rodrigues(R_wc)

                # Draw the axes at the workspace origin
                temp1 = cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvec_wc, t_wc, 50)
                # cv2.imshow('2', temp1)
                return avg_T_wc, temp1  # c = Twc * w,  c = RT_dc * d
            else:
                return None, None
        else:
            return None, None

def convert_to_right_hand(RT):
    R = RT[:3, :3]
    T = RT[:3, 3]
    

    R[:, 2] *= -1
    T[2] *= -1
    
    RT[:3, :3] = R
    RT[:3, 3] = T
    
    return RT


def spiral_search(d_frame, start_x, start_y, max_radius=10):
    # Initialize movement directions (right, down, left, up)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    step_count = 1
    current_direction_index = 0
    x, y = start_x, start_y
    steps_taken_in_current_leg = 0
    changes_in_direction = 0  # Count how many times we change direction

    while True:
        for _ in range(step_count):
            # Check boundaries
            if 0 <= x < 1920 and 0 <= y < 1080:
                if d_frame[y, x, 0] != 0:  # Check for valid depth
                    return x, y, d_frame[y, x]

            # Move to next position in the current direction
            x += directions[current_direction_index][0]
            y += directions[current_direction_index][1]

            # Check if we need to change direction
            steps_taken_in_current_leg += 1
            if steps_taken_in_current_leg == step_count:
                current_direction_index = (current_direction_index + 1) % 4
                steps_taken_in_current_leg = 0
                changes_in_direction += 1

                # Change the number of steps every two direction changes
                if changes_in_direction == 2:
                    step_count += 1
                    changes_in_direction = 0

        # Check if the search radius is exceeded
        if step_count > max_radius:
            break

    return None


def drawDepth(depth_image):
    depth_image = np.clip(depth_image, 50, 2000)

    depth_image = np.nan_to_num(depth_image, nan=50, posinf=2000, neginf=50)
    minVal = np.min(depth_image)
    maxVal = np.max(depth_image)


    if minVal == maxVal:
        scaled_depth = np.zeros(depth_image.shape, dtype=np.uint8)
    else:
        scaled_depth = ((depth_image - minVal) / (maxVal - minVal) * 255).astype(np.uint8)
    
    color_depth = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)
    # cv2.imshow('Color Depth Image', color_depth)
    # cv2.waitKey(0)

    return color_depth
