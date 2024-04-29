import cv2
import numpy as np
import os
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

class Params:
    def __init__(self) -> None:
        self.d_intrinsics = np.array([[362.4679, 0, 261.4258], [0, 361.1774, 208.1123], [0, 0, 1]])
        self.c_intrinsics = np.array([[1038.6, 0, 951.1293], [0, 1038.9, 526.1001], [0, 0, 1]])
        """ self.RT_d_to_c = np.array([[0.999820242867276, -0.005590505683472, 0.018117069272502, -51.337633677812520],
                           [0.005420373979096, 0.999940881177058, 0.009426223887326, 3.371261769890467],
                           [-0.018168695570907, -0.009326328165486, 0.999791437302901, 17.491786028014985],
                           [0, 0, 0,1]]) """       
        self.RT_d_to_c = np.array([[1, 0, 0, 3.37],
                           [0, 1, 0, -51.34],
                           [0, 0, 1, -17.49],
                           [0, 0, 0, 1]])
        self.d_distor = np.array([0.1174, 0.2216, 0, 0, 0])
        self.c_distor = np.array([0.0378, 0.1169, 0, 0, 0])


def d_To_c(x, y, z, param : Params):
    color_pixel_coordinate = np.array([[x],
                                    [y],
                                    [1]])
    d_world_coordinate = np.dot(np.linalg.inv(param.d_intrinsics),z * color_pixel_coordinate)
    d_world_coordinate = np.r_[d_world_coordinate, [[1]]]
    c_world_coordinate = np.dot(param.RT_d_to_c, d_world_coordinate)
    c_world_coordinate = np.delete(c_world_coordinate, 3, axis=0)
    c_pixel_coordinate = np.dot(param.c_intrinsics, c_world_coordinate)
    if c_pixel_coordinate[2, 0] != 0:
        c_pixel_coordinate = c_pixel_coordinate / c_pixel_coordinate[2, 0]
    return c_pixel_coordinate 
    
def d_2_c(x, y, z, param : Params):
    xw = (x - 212) / 367.816 * z
    yw = (y - 256) / 367.816 * z
    d_world_coordinate = np.array([[xw], [yw], [z], [1]])
    c_world_coordinate = np.dot(param.RT_d_to_c, d_world_coordinate)
    c_pixel_x = 1066.666666667 / z * c_world_coordinate[0,0] + 540
    c_pixel_y = 1066.666666667 / z * c_world_coordinate[1,0] + 960
    return c_pixel_x, c_pixel_y

def get_last_rbg():
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [1080, 1920, 4])[:, :, 0:3]

def get_last_depth():
    frame = kinect.get_last_infrared_frame()
    frame = frame.astype(np.uint8)
    dep_frame = np.reshape(frame, [424, 512])
    # np.savetxt('D:\\Desktop\\photo\\cancan.txt', dep_frame, fmt = "%d", delimiter = ' ')
    return cv2.cvtColor(dep_frame, cv2.COLOR_GRAY2RGB)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)


# cv2.namedWindow('c')
# cv2.namedWindow('d')
# cv2.namedWindow('e')
counter = 0
while 1:

    c_frame = kinect.get_last_color_frame()
    c_frame = np.reshape(c_frame, [1080, 1920, 4])[:, :, 0:3]
    c_frame = cv2.flip(c_frame, 1)
    combine_frame = c_frame.copy()
    temp_frame = np.zeros((1080, 1920))
    d_frame = kinect.get_last_depth_frame()

    d_frame = np.reshape(d_frame, [424, 512])
    d_frame = cv2.flip(d_frame, 1)
    
    dd = cv2.cvtColor(d_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # c_frame = cv2.undistort(c_frame, Params().c_intrinsics, Params().c_distor)
    # d_frame = cv2.undistort(d_frame, Params().d_intrinsics, Params().d_distor)
    for row in range(424):
        for col in range(512):
             if (d_frame[row][col] <= 50000 and d_frame[row][col]>0):
                u_f, v_f = d_2_c(row, col, d_frame[row][col], Params())
           
                temp = d_frame.astype(np.uint8)
                x = int(u_f)
                y = int(v_f)
                z = temp[row][col]
                # print(str(x) + ' ' + str(y) + ' ' + str(z))
                if not (x < 0 or x > 1079 or y > 1910 or y < 0):
                    temp_frame[x , y] = z
                    combine_frame[x, y] = z
                    dd[row, col] = c_frame[x, y]


            
    pathCom = 'D:\\Desktop\\photocom\\'
    os.makedirs(pathCom, exist_ok = True)        
    temp_frame = temp_frame.astype(np.uint8)
    cv2.imwrite(pathCom + str(counter) + 'c2d' + '.jpg', dd)
    cv2.imwrite(pathCom + str(counter) + '.jpg', cv2.cvtColor(temp_frame, cv2.COLOR_GRAY2RGB))
    cv2.imwrite(pathCom + str(counter) + 'rpg' + '.jpg', combine_frame)
    counter += 1

    # cv2.imshow('c', c_frame)
    # cv2.imshow('d', dd)
    # cv2.imshow('e', combine_frame)

# c_frame = cv2.imread('C:\\Users\\96156\\fifa\\fp\\photo\\color1\\12.jpg')
    
# d_frame = cv2.imread('C:\\Users\\96156\\fifa\\fp\\photo\\depth1\\12.jpg')

# d_frame = cv2.cvtColor(d_frame, cv2.COLOR_BGR2GRAY)

# d_frame = d_frame.astype(np.uint16)

# temp_frame = np.zeros((424, 512, 3))
# print(d_frame)

# for row in range(424):
#     for col in range(512):
#         if d_frame[row][col] <= 50000 and d_frame[row][col] > 0:
        
#             u_f, v_f = d_2_c(row, col, d_frame[row][col], Params())
#             x = int(u_f)
#             y = int(v_f)
#             if not (x < 0 or x > 1079 or y > 1910 or y < 0):
#                 temp_frame[row][col] = c_frame[x][y][0:3]
           
# cv2.imshow('e', temp_frame)
# cv2.waitKey(10000)
