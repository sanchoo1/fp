from ultralytics import YOLO
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# Load a  model
model = YOLO( ROOT / 'best736.pt')

# Open the video
source = 3
cap = cv2.VideoCapture(source)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz = 736)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("detect", annotated_frame)

        # View results
        for r in results:
            boxes = r.boxes
            if boxes.xyxy.numel() > 0:
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    print(x1, y1, x2, y2)

              # print the Boxes object containing the detection bounding boxes

        if cv2.waitKey(1) == 'q':
            break

