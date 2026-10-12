import queue
import threading

import config
from core.capture import CaptureThread
from core.detector import DetectorThread
from core.recognizer import RecognizerThread
from io_handlers.reference_store import ReferenceStore
from io_handlers.voice import VoiceThread


class FrasPipeline:
    """
    Capture -> Detector -> Recognizer -> result_queue
                                             |
                                    (main loop) -> Voice
    Every arrow is a small bounded queue, so slow stages drop old frames
    instead of building up lag.
    """

    def __init__(self):
        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=2)
        self.face_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.voice_queue = queue.Queue(maxsize=10)
        self.preview_queue = queue.Queue(maxsize=2) if config.SHOW_WINDOW else None

        store = ReferenceStore(config.REFERENCE_DIR)

        self.threads = [
            CaptureThread(self.frame_queue, self.stop_event,
                          config.CAMERA_INDEX, self.preview_queue),
            DetectorThread(self.frame_queue, self.face_queue, self.stop_event,
                           config.DETECTOR_PROTOTXT, config.DETECTOR_MODEL,
                           config.DETECTOR_CONFIDENCE),
            RecognizerThread(self.face_queue, self.result_queue, self.stop_event,
                             store, config.RECOGNITION_THRESHOLD),
            VoiceThread(self.voice_queue, self.stop_event, config.VOICE_COOLDOWN),
        ]

    def start(self):
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2)