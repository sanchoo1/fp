import time
import cv2
import numpy as np
import os
import cv2.aruco as aruco

# [8.201776125356415e+02,-0.085438805814383,3.294454919674360e+02;0,8.190036007627443e+02,2.431314948943652e+02;0,0,1]
# [0.002196074586884,0.703960778981213,-1.335740449221103]
# [9.770213101823787e-04,4.631756276924443e-04]

class Params:
    def __init__(self) -> None:
        self.intrinsics = np.array([[820.177613, -0.085439, 329.445492], [0, 819.003601, 243.131495], [0, 0, 1]])
        self.distor = np.array([0.002196, 0.703961, 0.000977, 0.000463, -1.335740])


def find_active_cameras(limit=10):
    # Check the first 'limit' indices.
    active_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW is added for DirectShow (Windows only)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                active_cameras.append(i)
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(500)  # Display each frame for 500 ms
                cv2.destroyAllWindows()
            cap.release()
        time.sleep(0.1)  # Give some time to release the camera properly

    print(active_cameras)

def generateMarker():
    # Generate ArUco Markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.generateImageMarker(dictionary, 23, 200, markerImage, 1)
    cv2.namedWindow('Frame')
    while True:
        cv2.imshow('Frame', markerImage)
        if cv2.waitKey(1) == ord('q'):
            break
    # cv2.imwrite("marker23.png", markerImage)

def create_marker_image(size, marker_id, dictionary_type=aruco.DICT_6X6_250):
    """ Generate an image of a single ArUco marker. """
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    img = aruco.generateImageMarker(aruco_dict, marker_id, size)
    return img

def generate_work_plane(image_size, marker_size, marker_ids):
    """
    Create a white image with Aruco markers at each corner.
    
    :param image_size: Size of the square image in pixels
    :param marker_size: Size of each marker in pixels
    :param marker_ids: List of four marker IDs for the corners
    :return: Image with Aruco markers
    """
    if len(marker_ids) != 4:
        raise ValueError("Four marker IDs are required.")
    
    # Create a white background image
    plane = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    
    # Marker positions (top-left corners)
    positions = [
        (0, 0),  # Top-left corner
        (image_size - marker_size, 0),  # Top-right corner
        (0, image_size - marker_size),  # Bottom-left corner
        (image_size - marker_size, image_size - marker_size)  # Bottom-right corner
    ]
    
    # Place each marker in one of the corners
    for marker_id, position in zip(marker_ids, positions):
        marker_image = create_marker_image(marker_size, marker_id)
        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR
        x, y = position
        plane[y:y+marker_size, x:x+marker_size] = marker_image
    
    return plane

def capture():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("cap not open")
        exit()

    cv2.namedWindow('a')
    counter = 0
    path = 'C:\\Users\\96156\\fifa\\pho\\'
    os.makedirs(path, exist_ok= True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('a', frame)
        if (counter % 200 == 0):
            cv2.imwrite(path + str(int(counter/200)) + '.jpg', frame)

        counter += 1

        cv2.imshow('a', frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

# # find_active_cameras()
# # capture()
# # generateMarker()

# # Parameters
# image_size = 800  # Size of the work plane image in pixels
# marker_size = 100  # Size of each Aruco marker in pixels
# marker_ids = [101, 102, 103, 104]  # Unique IDs for each marker

# # Generate the work plane image
# work_plane = generate_work_plane(image_size, marker_size, marker_ids)

# # Display the image
# cv2.imshow('Work Plane with Aruco Markers', work_plane)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Optionally, save the image to file
# # cv2.imwrite('work_plane_with_markers.png', work_plane)