from ultralytics import YOLO
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if __name__ == '__main__':

    # Load a model
    model = YOLO(ROOT / 'xs.pt')

    # Customize validation settings
    validation_results = model.val(data="xss/1.yaml", imgsz=1056)