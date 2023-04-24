import cv2
import numpy as np
from src.model import model
from src.config import OFFSET, iou_threshold, score_threshold


def detect(frame, detected_balls, num):
    res = model(frame, conf=score_threshold, iou=iou_threshold, verbose=False)[0]

    detections = []
    for i in range(len(res.boxes)):
        box = res.boxes[i]
        xywh = box.xywh[0]
        xyxy = box.xyxy[0]
        score = box.conf[0].item()

        centerX = xywh[0]
        centerY = xywh[1]

        print(
            f"Baseball Detected ({centerX}, {centerY}), Confidence: {str(round(score, 2))}, frame: {num}"
        )
        detected_balls.append([centerX, centerY])
        detections.append(
            np.array(
                [
                    xyxy[0] - OFFSET,
                    xyxy[1] - OFFSET,
                    xyxy[2] + OFFSET,
                    xyxy[3] + OFFSET,
                    score,
                ]
            )
        )

    return detections
