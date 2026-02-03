from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

YOLO_WEIGHTS = (
    BASE_DIR /
    "notebooks/runs/detect/runs/models/smartvision_yolo_10ep/weights/best.pt"
)

def load_yolo():
    return YOLO(str(YOLO_WEIGHTS))

def run_detection(model, image, conf):
    return model.predict(image, conf=conf, save=False)
