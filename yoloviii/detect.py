from ultralytics import YOLO
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# Load a  model
model = YOLO( ROOT / 'xxx.pt')

# Open the video
source = 0

# s = cv2.imread('C:/Users/96156/Desktop/hd.jpg')

# cropped = s[12:1068, 432:1488]

# results = model(cropped)
# annotated_frame = results[0].plot()

# while 1:
#     cv2.imshow("detect", annotated_frame)
#     if cv2.waitKey(1) == 'q':
#         break


cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("[ERR ] Failed to open video source.")
    exit()

#Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    cropped = frame[12:1068, 432:1488]


    if success:
        # Run YOLOv8 inference on the frame
        results = model(cropped)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("detect", annotated_frame)

        # View results
        # for r in results:
        #     boxes = r.boxes
        #     if boxes.xyxy.numel() > 0:
        #         for box in boxes.xyxy:
        #             x1, y1, x2, y2 = box.tolist()
        #             print(x1, y1, x2, y2)

              # print the Boxes object containing the detection bounding boxes

        if cv2.waitKey(1) == 'q':
            break

    else:
        print("err")
        break

cap.release()
cv2.destroyAllWindows()

