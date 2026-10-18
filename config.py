import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 0

DETECTOR_PROTOTXT = os.path.join(BASE_DIR, "models", "deploy.prototxt")
DETECTOR_MODEL = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
DETECTOR_CONFIDENCE = 0.5

REFERENCE_DIR = os.path.join(BASE_DIR, "reference_faces")
RECOGNITION_THRESHOLD = 0.6

VOICE_COOLDOWN = 25.0  # seconds before the same name is announced again

SHOW_WINDOW = True      # set False for headless Pi deployment
BOX_TTL = 0.5           # seconds to keep drawing the box after the last detection
# --- Blink challenge ---
AUTO_CHALLENGE = True         # False = only start with the 'c' key
BLINK_TARGET = 2
BLINK_TIMEOUT = 8.0
BLINK_PROMPT_DELAY = 1.5      # blinks while the voice is speaking are not counted
BLINK_RETRY_SECONDS = 5.0     # wait after a failed check before asking again
BLINK_CLOSE_THRESHOLD = 0.23
BLINK_OPEN_THRESHOLD = 0.25
BLINK_MIN_CLOSED_FRAMES = 2
BLINK_MAX_CLOSED_SECONDS = 0.5
LIVE_TTL = 15.0   # seconds a verified person stays verified after leaving the camera view