from ultralytics import YOLO

# Load a model
model = YOLO(
    "runs/detect/train2/weights/best.pt",
    task="detect",
)  # load a pretrained model (recommended for training)

# metrics = model.val()  # evaluate model performance on the validation set

success = model.export(
    format="saved_model"
)  # export the model to TF saved_model format
