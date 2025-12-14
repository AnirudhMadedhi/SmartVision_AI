
from ultralytics import YOLO

def load_yolo(weights='yolov8n.pt'):
    return YOLO(weights)
