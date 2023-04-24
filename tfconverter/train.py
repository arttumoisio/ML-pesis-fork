from ultralytics import YOLO
from roboflow import Roboflow

dataset = (
    Roboflow(api_key="ru3UdSz1sKekX9RgQTG9")
    .workspace("ppt-t3ll4")
    .project("ptt2")
    .version(4)
    .download("yolov8")
)

# Load a model
model = YOLO(
    # "yolov8/yolov8n.pt",
    # "yolov8/yolov8s.pt",
    # "yolov8/yolov8m.pt",
    "runs/detect/train2/weights/epoch32.pt",
    task="detect",
)  # load a pretrained model (recommended for training)

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    patience=15,
    batch=64,
    # device="mps",
    save_period=4,
    plots=True,
    shear=2,
)  # train the model
