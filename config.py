import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 0

DETECTOR_PROTOTXT = os.path.join(BASE_DIR, "models", "deploy.prototxt")
DETECTOR_MODEL = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
DETECTOR_CONFIDENCE = 0.5

REFERENCE_DIR = os.path.join(BASE_DIR, "reference_faces")
RECOGNITION_THRESHOLD = 0.6

VOICE_COOLDOWN = 10.0   # seconds before the same name is announced again

SHOW_WINDOW = True      # set False for headless Pi deployment
BOX_TTL = 0.5           # seconds to keep drawing the box after the last detection