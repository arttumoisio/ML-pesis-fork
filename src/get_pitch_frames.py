import copy
import cv2
import numpy as np
from src.FrameInfo import FrameInfo
from src.utils import distance, fill_lost_tracking
from src.SORT_tracker.sort import Sort
from src.colors.colors import track_colors
from src.detect_ball import detect


# Get the pitching section in the whole video
def get_pitch_frames(video_path, infer, input_size, iou, score_threshold):
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Store the pitching section in pitch_frames
    pitch_frames = []
    detected_balls = []
    tracked_balls = []
    frames = []
    tracker_min_hits = 1
    frame_id = 0
    tracker_iou_threshold = 0.3

    # Create Object Tracker
    tracker = Sort(
        max_age=8, min_hits=tracker_min_hits, iou_threshold=tracker_iou_threshold
    )

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(FrameInfo(frame, False))
        else:
            print("Processing complete")
            break

        # Detect the baseball in the frame
        detections = detect(
            infer, frame, input_size, iou, score_threshold, detected_balls, frame_id
        )

        # Feed in detections to obtain SORT tracking
        if len(detections) > 0:
            trackings = tracker.update(np.array(detections))
        else:
            trackings = tracker.update()

        # Add the valid trackings to balls_list
        for t in trackings:
            t = [int(i) for i in t]

            color = track_colors[t[4] % 12]
            centerX = int((t[0] + t[2]) / 2)
            centerY = int((t[1] + t[3]) / 2)
            tracked_balls.append([centerX, centerY, color])

        # Store the frames with ball tracked
        if len(trackings) > 0:
            # Only run at the first track from SORT
            if len(pitch_frames) == 0:
                last_tracked_frame = frame_id
                add_balls_before_SORT(
                    frames, detected_balls, tracked_balls, tracker_min_hits
                )
                # Add prior 20 frames before the first balsadl
                pitch_frames.extend(frames[-20:])

            # Add lost frames if any
            add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames)

            # Append the frame with detected ball location
            last_ball = tuple(tracked_balls[-1][:-1])
            pitch_frames.append(FrameInfo(frame, True, last_ball, color))
            last_tracked_frame = frame_id

        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

        frame_id += 1

    # Use Polyfit to approximate the untracked balls
    fill_lost_tracking(pitch_frames)

    # Add five more frames after the last tracked frame
    pitch_frames.extend(frames[last_tracked_frame : last_tracked_frame + 10])
    return pitch_frames, width, height, fps


def add_balls_before_SORT(frames, detected, tracked, tracker_min_hits):
    distance_threshold = 100
    first_ball = tracked[0]
    color = first_ball[2]
    balls_to_add = []

    # Get the untracked balls that's close enough to the first tracked ball
    for untracked in detected[-(tracker_min_hits + 1) :]:
        if distance(untracked, first_ball) < distance_threshold:
            untracked.append(color)
            balls_to_add.append(untracked)

    # Add the untracked balls to frame
    modify_frames = frames[-(tracker_min_hits + 1) :]
    balls_to_add_temp = copy.deepcopy(balls_to_add)

    for point in balls_to_add_temp:
        del point[2]
    balls_to_add_temp = np.array(balls_to_add_temp, dtype="int32")

    for idx, frame in enumerate(modify_frames):
        frames[-((tracker_min_hits + 1) - idx)] = FrameInfo(
            frame.frame, True, tuple(balls_to_add_temp[idx]), color
        )


def add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames):
    if frame_id - last_tracked_frame > 1:
        print("Lost frames:", frame_id - last_tracked_frame)
        frames_to_add = frames[last_tracked_frame:frame_id]

        # Mark the detection in lost in this frame
        for ball_frame in frames_to_add:
            ball_frame.ball_lost_tracking = True
        pitch_frames.extend(frames_to_add)
