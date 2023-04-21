from ultralytics import YOLO
from roboflow import Roboflow

dataset = (
    Roboflow(api_key="ru3UdSz1sKekX9RgQTG9")
    .workspace("ppt-t3ll4")
    .project("ptt2")
    .version(3)
    .download("yolov8")
)

# Load a model
model = YOLO(
    "yolov8/yolov8n.pt",
    # "model/yolov8/yolov8s.pt"
    # "model/yolov8/yolov8m.pt"
    task="detect",
)  # load a pretrained model (recommended for training)

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=1,
    imgsz=640,
    patience=4,
    batch=64,
    # device="mps",
    save_period=4,
    plots=True,
)  # train the model
