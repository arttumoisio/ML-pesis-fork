from datetime import datetime

modelPath = (
    # "runs/detect/train/weights/best.pt"
    "runs/detect/train2/weights/epoch24.pt"
)
# Detect config
PADDING = 150  # maybe should be % of video height 10 = 108 with full hd 1920x1080
iou_threshold = 0.2
score_threshold = 0.4

# SORT config
tracker_min_hits = 4
tracker_iou_threshold = 0.2
max_age = 14

pathStart = "Overlay"
f = f"/{pathStart}_{datetime.today().strftime('%H%M')}_"
e = f".mp4"
outputPath = (
    ## spacing
    f"{f}{PADDING}_{iou_threshold}_{score_threshold}_{tracker_min_hits}_{tracker_iou_threshold}_{max_age}{e}"
)

# Basic conf
# DEFAULT_VIDEOS_FOLDER = "./videos/lot"
DEFAULT_VIDEOS_FOLDER = "./videos/s"

# Overlay config
fps_percentage = 0.25  # 1 = full fps, 0.5 = half
base_frame_weight = 0.45  # orig 0.55

ball_size = 4  # TODO count from bbox heights
line_thickness = round(1.2 * ball_size)  # TODO count from ball_size
trajectory_weight = 0.6  # orig 0.7


before_frames = 30
after_frames = 50
