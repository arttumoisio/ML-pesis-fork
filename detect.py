from ultralytics import YOLO
from roboflow import Roboflow
import torch

dataset = (
    Roboflow(api_key="ru3UdSz1sKekX9RgQTG9")
    .workspace("ppt-t3ll4")
    .project("ptt2")
    .version(1)
    .download("yolov8")
)

# Load a model
model = YOLO(
    "model/yolov8/yolov8m.pt"
)  # load a pretrained model (recommended for training)

model.train(
    task="detect",
    data=f"{dataset.location}/data.yaml",
    epochs=12,
    imgsz=720,
    patience=4,
    batch=-1,
    device="mps",
    workers=8,
    save_period=4,
)  # train the model

metrics = model.val()  # evaluate model performance on the validation set

success = model.export(
    format="saved_model"
)  # export the model to TF saved_model format
