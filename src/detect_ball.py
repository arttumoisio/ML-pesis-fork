import cv2
import numpy as np
from src.model import model
from src.config import PADDING, iou_threshold, score_threshold


def get_detections_in_format(boxes, detected_balls):
    detections = []
    for box in boxes:  # Indexing boxes object gives boxes object
        xywh = box.xywh[0]
        xyxy = box.xyxy[0]
        score = box.conf[0].item()

        centerX = xywh[0]
        centerY = xywh[1]

        detected_balls.append([centerX, centerY])
        detections.append(
            np.array(
                [
                    xyxy[0].cpu() - PADDING,
                    xyxy[1].cpu() - PADDING,
                    xyxy[2].cpu() + PADDING,
                    xyxy[3].cpu() + PADDING,
                    score,
                ]
            )
        )
    if len(detections) > 0:
        return np.array(detections)
    else:
        return np.empty((0, 5))
