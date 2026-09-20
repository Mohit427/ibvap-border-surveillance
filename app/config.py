"""Central paths and tunables for the IBVAP prototype."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
UPLOADS_DIR = DATA_DIR / "uploads"
STATIC_DIR = ROOT_DIR / "static"

YOLO_DETECTION_WEIGHTS = MODELS_DIR / "yolov8n.pt"
PLATE_DETECTOR_WEIGHTS = MODELS_DIR / "plate_detector.pt"
FACE_DETECTOR_WEIGHTS = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

# Default sample clip shown when no webcam and no upload is available.
DEFAULT_SAMPLE_VIDEO = SAMPLE_VIDEOS_DIR / "pedestrian_crossing.mp4"

# --- Processing / performance ---
# The whole pipeline (detection + ANPR + face + tracking) is CPU-bound.
# We downscale every frame before running any model and cap the processing
# rate, trading frame rate for running every module in real time on a
# laptop CPU. Increase PROCESS_WIDTH / TARGET_FPS on faster machines.
PROCESS_WIDTH = 640
TARGET_FPS = 10
JPEG_QUALITY = 75

# Run the (slow) ANPR plate-detector + OCR only every N processed frames.
ANPR_FRAME_INTERVAL = 15

# --- Detection ---
DETECTION_CONF = 0.35
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACKED_CLASS_IDS = [PERSON_CLASS_ID, *VEHICLE_CLASS_IDS.keys()]

# --- Fence / activity heuristics (in processed-frame pixel space) ---
TRAIL_HISTORY_SECONDS = 15
LOITER_WINDOW_SECONDS = 8.0
LOITER_RADIUS_PX = 45
FAST_MOVEMENT_PX_PER_SEC = 260
ALERT_COOLDOWN_SECONDS = 12.0

# --- Night mode ---
NIGHT_BRIGHTNESS_THRESHOLD = 70

# --- Face ---
FACE_SCORE_THRESHOLD = 0.7

# --- ANPR ---
PLATE_DETECT_CONF = 0.35
