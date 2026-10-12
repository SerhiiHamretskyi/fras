import threading
import queue
import cv2 as cv


class CaptureThread(threading.Thread):
    """
    Owns the camera. Pushes the latest frame into frame_queue (and into
    preview_queue if given). When a queue is full, the oldest frame is
    dropped so we always work with the freshest one.
    """

    def __init__(self, frame_queue: queue.Queue, stop_event: threading.Event,
                 camera_index: int = 0, preview_queue: queue.Queue = None):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.preview_queue = preview_queue
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.cap = None

    @staticmethod
    def _push(q: queue.Queue, frame):
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(frame)

    def run(self):
        self.cap = cv.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Camera is not available")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end). Exiting ...")
                self.stop_event.set()
                break

            self._push(self.frame_queue, frame)
            if self.preview_queue is not None:
                self._push(self.preview_queue, frame)

        self.cap.release()