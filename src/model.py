from ultralytics import YOLO
from src.config import modelPath

# Load a model
model = YOLO(
    modelPath,
    # "tfconverter/runs/detect/train/weights/best.pt",
    # "tfconverter/runs/detect/train2/weights/epoch12.pt",
    task="detect",
)
