from ultralytics import YOLO
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if __name__ == '__main__':
    model = YOLO(ROOT/ 'best.pt')

    # model = YOLO('yolov8s')
    results = model.train(data = 'xss/1.yaml',  epochs = 300, imgsz = 1056, resume = True, device = 0)
