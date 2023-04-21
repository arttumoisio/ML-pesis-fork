import cv2
import numpy as np
import tensorflow as tf

OFFSET = 100
ACCURACY = 0.95


# Tensorflow Object Detection API Sample
def detect(infer, frame, input_size, iou, score_threshold, detected_balls, num):
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for _, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score_threshold,
    )

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    valid_detections = valid_detections.numpy()

    frame_h, frame_w, _ = frame.shape
    detections = []

    for i in range(valid_detections[0]):
        score = scores[0][i]
        if score > ACCURACY:
            coor = boxes[0][i]
            coor[0] = coor[0] * frame_h
            coor[2] = coor[2] * frame_h
            coor[1] = coor[1] * frame_w
            coor[3] = coor[3] * frame_w

            centerX = int((coor[1] + coor[3]) / 2)
            centerY = int((coor[0] + coor[2]) / 2)

            print(
                f"Baseball Detected ({centerX}, {centerY}), Confidence: {str(round(score, 2))}, frame: {num}"
            )
            # cv2.circle(frame, (centerX, centerY), 15, (255, 0, 0), -1)
            detected_balls.append([centerX, centerY])
            detections.append(
                np.array(
                    [
                        coor[1] - OFFSET,
                        coor[0] - OFFSET,
                        coor[3] + OFFSET,
                        coor[2] + OFFSET,
                        score,
                    ]
                )
            )

    return detections
