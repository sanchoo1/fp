import cv2
import numpy as np
from threading import Thread, Event
import time
from pykinect2 import PyKinectV2, PyKinectRuntime
from pykinect2.PyKinectV2 import *
from queue import Queue
import os
import mapper


import util

last_time = time.time()
mapping = False
detect = False

def aruc(aruco, stat_dict, c_frame, stopEvent):
    stat_dict['markers'] = 0
    global last_time
    while not stopEvent.is_set():
        if c_frame is not None:
            corners, _, frame = aruco.detected(c_frame.copy())             
            stat_dict['markers'] = len(corners)
            text = f"{stat_dict['markers']} marker{'s' if stat_dict['markers'] != 1 else ''} being detected"
            org = (50, 50)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)
            if stat_dict['markers'] >= 1:
                RT, c_frame = aruco.get_camera_pose_from_aruco_markers(c_frame.copy())
                stat_dict['rt'] = RT
                currunt_time = time.time()
                if currunt_time - last_time > 10:
                    # print(RT)
                    last_time = currunt_time
            return c_frame

def ON_EVENT_LBUTTONDOWN(event, y, x, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        c, dd, mapd, stat_dict = param
        k = stat_dict.get("camera", None)
        red_point = [y, x]
        radius = 5 
        color = (0, 0, 255)  # red
        thickness = -1  # fill
        nearest = util.spiral_search(mapd,red_point[0], red_point[1])
        if nearest:
            print("color pixel : ", red_point)
            dpoint = nearest[2]
            print("map pixel : ", nearest[0:2])
            dpoint = dpoint[1:]
            # print(dpoint)
            cv2.circle(dd, dpoint, radius, color, thickness) # draw a red point
            cv2.circle(c, red_point, radius, color, thickness)
            dcam = k.getCameraCoo(nearest[2])
            print(dcam)

            if stat_dict['markers'] > 0:
                rt = stat_dict.get("rt", None)
                print(rt)
                if rt is not None:
                    print("world coordinate:", k.getPosition(dcam, rt))


def display(frame_queue, stat_dict, stopEvent):

    cv2.namedWindow("color", cv2.WINDOW_NORMAL)  
    cv2.namedWindow("MyDepth", cv2.WINDOW_NORMAL)
    if mapping:
        cv2.namedWindow("map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("map", 1920, 1080)

    d = None
    c = None
    map = None
    mapd = None
    an = None
    frame_data = None
    previous_time = time.time()
    frame_intervals = []
    if detect:
        cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    with open('frame_intervals.txt', 'w') as f:
        while not stopEvent.is_set():
            while not frame_queue.empty():
                frame_data = frame_queue.get_nowait()
                current_time = time.time()
                interval = current_time - previous_time
                frame_intervals.append(interval)
                # print(interval)
                f.write(f'{interval}\n')
                previous_time = current_time
            if frame_data:
                d = frame_data['depth']
                c = frame_data['color']
                if detect:
                    an = frame_data['detect']
                    cv2.imshow("detect", an)
                if mapping:
                    mapd = frame_data['mapD']
                    mapdd = frame_data['mapdd']

                    cv2.setMouseCallback("color", ON_EVENT_LBUTTONDOWN, param=[c, d, mapd, stat_dict]) 
                    cv2.imshow("map", mapdd)
                
                if d is not None and c is not None:
                    cv2.imshow("MyDepth", d)
                    cv2.imshow("color", c)
        
            if cv2.waitKey(20) & 0xFF == ord('q'):
                stopEvent.set()

def main():
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    k = util.MyKinect()
    aruco = util.MyAruco(k)
    yolo = util.Yolo8()

    frame_queue = Queue()
    stat_dict = {}
    stat_dict['camera'] = k

    calibrate = True
   
    Display = True

    stopEvent = Event()
    global last_time
    last_time = time.time()

    global mapping
    mapping = True

    global detect
    detect = False

    display0_thread = Thread(target=display, args=(frame_queue, stat_dict, stopEvent))

    an = None
    dd = None
    drawMapDepth = None

    try:
        while not stopEvent.is_set():
            if kinect.has_new_depth_frame():

                c_frame = kinect.get_last_color_frame()
                c_frame = c_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))[:, :, 0:3]
                c_frame = cv2.flip(c_frame, 1)

                d_frame = kinect.get_last_depth_frame()
                d_frame = d_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width))
                d_frame = cv2.flip(d_frame, 1)

                # red_point = [800, 500]
                
                # dd, drawMapDepth = k.generateD(d_frame)
                # ddd, _ = k.generateD2(d_frame)
                # d_frame = util.drawDepth(d_frame)
                # radius = 5 
                # color = (0, 0, 255)  # red
                # thickness = -1  # fill
                # nearest = util.spiral_search(dd,red_point[0], red_point[1])
                # nearest2 = util.spiral_search(ddd,red_point[0], red_point[1])
                # cv2.circle(c_frame, red_point, 8, color, thickness)
                # if nearest:
                #     print("color pixel : ", red_point)
                #     dpoint = nearest[2]
                #     dpoint2 = nearest2[2]
                #     point = dpoint[1:]
                #     point2 = dpoint2[1:]
                #     print(point, point2)
                #     text = f"({point[0]}, {point[1]})"
                #     cv2.putText(d_frame, text, (point2[0] + 10, point2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                #     # print(dpoint)
                #     cv2.circle(d_frame, point2, radius, color, thickness)
                #     cv2.circle(d_frame, point, radius, color, thickness) # draw a red point

                    # dpoint = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, red_point)
                    # text = f"({dpoint[0]}, {dpoint[1]})"
                    # cv2.putText(depth_img, text, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.circle(depth_img, dpoint, radius, color, thickness)

                    # dcam = util.getCameraCoo(nearest[2])

                if detect:
                    an = yolo.annotate(c_frame.copy())
                
                if calibrate:
                    c_frame = aruc(aruco, stat_dict, c_frame, stopEvent)

                if mapping:
                    dd, drawMapDepth = k.generateD(d_frame)

                d_frame = util.drawDepth(d_frame)
                frame_queue.put({'color': c_frame, 'depth': d_frame, 'mapD': dd, 'mapdd': drawMapDepth, 'detect': an})

                if Display:
                    display0_thread.start()
                    Display = False

    except KeyboardInterrupt:
        stopEvent.set()

    display0_thread.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()