import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Camera ---
CAMERA_INDEX = 0

# --- Face detector ---
DETECTOR_PROTOTXT = os.path.join(BASE_DIR, "models", "deploy.prototxt")
DETECTOR_MODEL = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
DETECTOR_CONFIDENCE = 0.5

# --- Recognition ---
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_faces")
RECOGNITION_THRESHOLD = 0.6

# --- Voice ---
VOICE_COOLDOWN = 25.0   # seconds before the same name is announced again

# --- Display ---
SHOW_WINDOW = True      # set False for headless Pi deployment
BOX_TTL = 0.5           # seconds to keep drawing the box after the last detection

# --- Liveness challenge ---
AUTO_CHALLENGE = True         # False = only start with the 'c' key
LIVE_TTL = 15.0               # seconds a verified person stays verified after leaving the view
BLINK_RETRY_SECONDS = 5.0     # wait after a failed check before asking again

# Step 1: blinks
BLINK_TARGET = 2
BLINK_TIMEOUT = 8.0
BLINK_PROMPT_DELAY = 1.5      # blinks while the voice is speaking are not counted
BLINK_CLOSE_THRESHOLD = 0.23
BLINK_OPEN_THRESHOLD = 0.25
BLINK_MIN_CLOSED_FRAMES = 2
BLINK_MAX_CLOSED_SECONDS = 0.5

# Step 2: random head movement (left / right / up / down)
HEAD_MOVES = 2              # how many random head commands per challenge, 0 = blinks only
HEAD_TIMEOUT = 6.0            # seconds to do each move
HEAD_PROMPT_DELAY = 1.0       # movement while the voice is speaking is not counted
HEAD_YAW_DELTA = 0.12         # left/right: how far the nose must shift (ratio)
HEAD_PITCH_DELTA = 0.08       # up/down
HEAD_HOLD_RESULTS = 2         # results in a row the head must stay in position

# Step 3: spoof (photo / screen) classifier, scored during the blink stage
SPOOF_MODEL = os.path.join(BASE_DIR, "models", "spoof_model.onnx")
SPOOF_THRESHOLD = 0.85        # average real-score needed to pass
SPOOF_ENABLED = True          # False = no spoof check, handy for testing