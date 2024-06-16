import cv2
import numpy as np
from threading import Thread, Event
import util  # Assuming util contains MyKinect, MyAruco, Yolo8, and drawDepth

def capture(kinect, frame_dict, stopEvent):
    while not stopEvent.is_set():
        c_frame = kinect.get_color_frame()
        d_frame = kinect.get_depth_frame()
        
        if c_frame is not None and d_frame is not None:
            frame_dict['color'] = c_frame
            frame_dict['depth'] = d_frame

def display(frame_dict, stopEvent):
    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("color", [1920, 1080])   
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("depth", [512, 424]) 
    
    while not stopEvent.is_set():
        d = frame_dict.get("depth", None)
        c = frame_dict.get("color", None)
        
        if d is not None:
            dd = util.drawDepth(d)
            cv2.imshow("depth", dd)
        
        if c is not None:
            cv2.imshow("color", c)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            stopEvent.set()
    
    cv2.destroyAllWindows()

def aruc(aruco, stat_dict, frame_dict, stopEvent):
    while not stopEvent.is_set():
        # Add your aruco processing code here
        pass

def main():
    kinect = util.MyKinect()
    aruco = util.MyAruco(kinect)
    yolo = util.Yolo8()

    frame_dict = {}
    stat_dict = {}
    stat_dict['pos'] = False

    stopEvent = Event()

    capture_thread = Thread(target=capture, args=(kinect, frame_dict, stopEvent))
    display_thread = Thread(target=display, args=(frame_dict, stopEvent))
    cali_thread = Thread(target=aruc, args=(aruco, stat_dict, frame_dict, stopEvent))

    capture_thread.start()
    display_thread.start()
    cali_thread.start()

    try:
        while not stopEvent.is_set():
            pass
    except KeyboardInterrupt:
        stopEvent.set()

    capture_thread.join()
    display_thread.join()
    cali_thread.join()

if __name__ == "__main__":
    main()