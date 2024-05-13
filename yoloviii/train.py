from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    results = model.train(data = 'dd/data.yaml', epochs = 300, imgsz = 720)