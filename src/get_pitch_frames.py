import copy
import cv2
import numpy as np
from src.FrameInfo import FrameInfo
from src.Laatu import Laatu
from src.utils import distance, fill_lost_tracking, get_laatu
from src.SORT_tracker.sort import Sort
from src.colors.colors import track_colors
from src.detect_ball import get_detections_in_format
from src.model import model
from src.config import (
    max_age,
    tracker_min_hits,
    tracker_iou_threshold,
    before_frames,
    after_frames,
    score_threshold,
    iou_threshold,
    PADDING,
    distance_threshold,
)


# Get the pitching section in the whole video
def get_pitch_frames(video_path):
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)
    laatu = get_laatu(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Store the pitching section in pitch_frames
    pitch_frames = []
    detected_balls = []
    tracked_balls = []
    frames = []

    # Detect the baseball in the video
    # results = model.track(
    # tracker="src/SORT_tracker/bot_custom_sort.yaml",
    results = model(
        source=video_path,
        conf=score_threshold,
        iou=iou_threshold,
        stream=True,
        # verbose=False,
    )

    # Create Object Tracker
    tracker = Sort(
        max_age=max_age, min_hits=tracker_min_hits, iou_threshold=tracker_iou_threshold
    )

    for frame_id, res in enumerate(results):
        frame = res.orig_img
        frames.append(FrameInfo(frame, False, laatu=laatu))

        # Feed in detections to obtain SORT tracking
        trackings = tracker.update(get_detections_in_format(res.boxes, detected_balls))

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
                add_balls_before_SORT(frames, detected_balls, tracked_balls, laatu)
                # Add prior frames before the first ball
                pitch_frames.extend(frames[-before_frames:])

            # Add lost frames if any
            add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames)

            # Append the frame with detected ball location
            last_ball = tuple(tracked_balls[-1][:-1])
            pitch_frames.append(FrameInfo(frame, True, last_ball, color, laatu=laatu))
            last_tracked_frame = frame_id

    print("Processing complete")

    # Use Polyfit to approximate the untracked balls
    fill_lost_tracking(pitch_frames)

    # Add five more frames after the last tracked frame
    pitch_frames.extend(frames[last_tracked_frame : last_tracked_frame + after_frames])
    return pitch_frames, width, height, fps


def add_balls_before_SORT(frames, detected, tracked, laatu: Laatu):
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
    balls_to_add_temp = balls_to_add_temp

    for idx, frame in enumerate(modify_frames):
        frames[-((tracker_min_hits + 1) - idx)] = FrameInfo(
            frame.frame, True, tuple(balls_to_add_temp[idx]), color, laatu=laatu
        )


def add_lost_frames(frame_id, last_tracked_frame, frames, pitch_frames):
    if frame_id - last_tracked_frame > 1:
        print("Lost frames:", frame_id - last_tracked_frame)
        frames_to_add = frames[last_tracked_frame:frame_id]

        # Mark the detection in lost in this frame
        for ball_frame in frames_to_add:
            ball_frame.ball_lost_tracking = True
        pitch_frames.extend(frames_to_add)
