from ultralytics import YOLO
from roboflow import Roboflow

if __name__ == '__main__':
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
        "yolov8/yolov8m.pt", # in use 24.4.2023 illalla
        # "yolov8/yolov8l.pt",
        # "yolov8/yolov8x.pt",
        task="detect",
    )  # load a pretrained model (recommended for training)

    
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=300,
        imgsz=800,
        patience=300,
        # batch=-1,
        batch=5,
        device=0,
        save_period=100,
        # plots=True,
    )  # train the model
