
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import C2D
import cv2

import aruco

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
            if 0 <= y < 1920 and 0 <= x < 1080:
                if d_frame[x, y, 0] != 0:  # Check for valid depth
                    return x, y, d_frame[x, y]

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

def getCameraCoo(array):
    z, x, y = array.toList()
    xw = (x - 212) / 367.816 * z
    yw = (y - 256) / 367.816 * z
    d_camera_coordinate = np.array([[xw], [yw], [z], [1]])
    return d_camera_coordinate

def ON_EVENT_LBUTTONDOWN(event, y, x, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        dd = C2D.generateD(param)
        nearest = spiral_search(dd, x, y)
        if nearest:
            # c = getCameraCoo(nearest[2])
            # print(c)
            print(nearest)

def main():
    para = C2D.Params()
    cameraMatrix = para.c_intrinsics
    distCoeffs = para.c_distor
    k = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

    cv2.namedWindow("1", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while 1:
        c_frame = k.get_last_color_frame()
        c_frame = np.reshape(c_frame, [1080, 1920, 4])[:, :, 0:3]
        c_frame = cv2.flip(c_frame, 1)
        d_frame = k.get_last_depth_frame()
        d_frame = np.reshape(d_frame, [424, 512])
        d_frame = cv2.flip(d_frame, 1)

        cv2.imshow("1", c_frame)
        cv2.resizeWindow("1", 1920, 1080)



        # corners, ids, rejectedImgPoints = aruco.detector.detectMarkers(c_frame)
        # if len(corners) > 0:  # at least one marker detected

        # t = aruco.get_camera_pose_from_aruco_markers(aruco.dictionary, aruco.parameters, c_frame, cameraMatrix, distCoeffs, 0.032)
        # paramm = [t, d_frame]
        cv2.setMouseCallback("1", ON_EVENT_LBUTTONDOWN, d_frame)
        if cv2.waitKey(10) == 27:  # ESC
            break
            



if __name__ == "__main__":
    main()