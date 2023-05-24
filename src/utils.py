import numpy as np
from src.Laatu import Laatu


class NoFramesException(Exception):
    pass


def fill_lost_tracking(frame_list):
    if len(frame_list) < 1:
        raise NoFramesException("No frames")

    balls_x = [frame.ball[0] for frame in frame_list if frame.ball_in_frame]
    balls_y = [frame.ball[1] for frame in frame_list if frame.ball_in_frame]

    balls_x = [b if isinstance(b, int) else b.cpu() for b in balls_x]
    balls_y = [b if isinstance(b, int) else b.cpu() for b in balls_y]

    # Get the polynomial equation
    curve = np.polyfit(balls_x, balls_y, 2)
    poly = np.poly1d(curve)

    lost_sections = []
    in_lost = False
    frame_count = 0

    # Get the sections where the ball is lost tracked
    for idx, frame in enumerate(frame_list):
        if frame.ball_lost_tracking and frame_count == 0:
            in_lost = True
            lost_sections.append([])

        if in_lost and not (frame.ball_lost_tracking):
            in_lost = False
            frame_count = 0

        if in_lost:
            lost_sections[-1].append(idx)
            frame_count += 1

    # Modify the frames in lost section with the approximated ball
    for lost_section in lost_sections:
        if lost_section:
            prev_frame = frame_list[lost_section[0] - 1]
            last_frame = frame_list[lost_section[-1] + 1]
            color = prev_frame.ball_color

            lost_idx = [frame_list[i] for i in lost_section]

            # Speed is the x difference for each frame
            diff = last_frame.ball[0] - prev_frame.ball[0]
            speed = int(diff / (len(lost_idx) + 1))

            for idx, frame in enumerate(lost_idx):
                x = prev_frame.ball[0] + (speed * (idx + 1))
                x = x if isinstance(x, int) else x.cpu()
                y = int(poly(x))
                frame.ball_in_frame = True
                frame.ball = (x, y)
                frame.ball_color = color
                # print('Fill', x, y)


def distance(x, y):
    temp = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return temp ** (0.5)


def get_laatu(path: str) -> Laatu:
    if (
        path.endswith("v.mp4")
        or path.endswith("V.mp4")
        or path.endswith("m.mp4")
        or path.endswith("M.mp4")
    ):
        return Laatu.VÄÄRÄ
    elif path.endswith("o.mp4") or path.endswith("O.mp4"):
        return Laatu.OIKEA
    elif path.endswith("l.mp4") or path.endswith("L.mp4"):
        return Laatu.LYÖTY
    elif path.endswith("t.mp4") or path.endswith("T.mp4"):
        return Laatu.TOLPPA
    elif path.endswith("p.mp4") or path.endswith("P.mp4"):
        return Laatu.PUOLIKAS
    elif path.endswith("m.mp4") or path.endswith("M.mp4"):
        return Laatu.MITÄTÖN
    else:
        return Laatu.UNKNOWN
