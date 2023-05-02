from ultralytics import YOLO
from roboflow import Roboflow

version = 5

if __name__ == "__main__":
    dataset = (
        Roboflow(api_key="ru3UdSz1sKekX9RgQTG9")
        .workspace("ppt-t3ll4")
        .project("ptt2")
        .version(version)
        .download("yolov8")
    )

    # Load a model
    model = YOLO(
        "yolov8/yolov8n.pt",
        # "yolov8/yolov8s.pt",
        # "yolov8/yolov8m.pt", # in use 24.4.2023 illalla
        # "yolov8/yolov8l.pt",
        # "yolov8/yolov8x.pt",
        task="detect",
    )  # load a pretrained model (recommended for training)

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=300,
        imgsz=640,
        patience=300,
        batch=-1, # auto batch
        # batch=32,
        device=0,  # 0 = cuda, null = cpu
        save_period=100,
    )  # train the model
