import cv2
import copy
import numpy as np
from src.FrameInfo import FrameInfo
from src.Laatu import Laatu
from src.config import ball_size, line_thickness, trajectory_weight


def draw_ball_curve(frame, trajectory):
    temp_frame = frame.copy()

    if len(trajectory):
        ball_points = copy.deepcopy(trajectory)
        highest = trajectory[0]
        for i, point in enumerate(ball_points):
            traj_color = point[2]
            if point[1] < highest[1]:  # if higher, replace
                highest = trajectory[i]
            del point[2:]
        ball_points = [
            [coord if isinstance(coord, int) else coord.cpu().item() for coord in ball]
            for ball in ball_points
        ]
        ball_points = np.array(ball_points, dtype="int32")
        # cv2.polylines(
        #     temp_frame,
        #     [ball_points],
        #     False,
        #     traj_color,
        #     line_thickness,
        #     lineType=cv2.LINE_AA,
        # )
        # frame = cv2.addWeighted(
        #     temp_frame, trajectory_weight, frame, 1 - trajectory_weight, 0
        # )

        last_frame = trajectory[-1]
        ball_color = get_ball_color(last_frame[3])
        last_ball = (
            last_frame[0]
            if isinstance(last_frame[0], int)
            else int(last_frame[0].item()),
            last_frame[1]
            if isinstance(last_frame[1], int)
            else int(last_frame[1].item()),
        )

        highest_ball = (
            highest[0] if isinstance(highest[0], int) else int(highest[0].item()),
            highest[1] if isinstance(highest[1], int) else int(highest[1].item()),
        )

        cv2.circle(frame, highest_ball, ball_size, ball_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            frame, highest_ball, ball_size, (0, 0, 0), 1, lineType=cv2.LINE_AA
        )  # Outline
        # cv2.circle(frame, last_ball, ball_size, (195, 195, 195), -1, lineType=cv2.LINE_AA)
        # cv2.circle(frame, last_ball, ball_size, (0, 0, 0), 1, lineType=cv2.LINE_AA) # Outline

    return frame


def get_ball_color(laatu: Laatu):  # -> tuple[int,int,int]:
    if laatu == Laatu.VÄÄRÄ:
        return (0, 0, 255)
    elif laatu == Laatu.OIKEA:
        return (102, 255, 0)
    # elif laatu == Laatu.LYÖTY:
    # elif laatu == Laatu.TOLPPA:
    # elif laatu == Laatu.PUOLIKAS:
    # elif laatu == Laatu.MITÄTÖN:
    # elif laatu == Laatu.UNKNOWN:
    else:
        return (255, 255, 255)
