import threading
import queue
import time
import cv2 as cv

RETRY_SECONDS = 10.0  # keep trying this long after the camera stops delivering frames


class CaptureThread(threading.Thread):
    """
    Owns the camera. Pushes the latest frame into frame_queue (and into
    preview_queue if given). When a queue is full, the oldest frame is
    dropped so we always work with the freshest one.

    If the camera stops delivering frames, it is closed and reopened for up
    to RETRY_SECONDS before the whole program is told to stop.
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

        failed_since = None
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if ret:
                failed_since = None
                self._push(self.frame_queue, frame)
                if self.preview_queue is not None:
                    self._push(self.preview_queue, frame)
                continue

            # No frame: close and reopen the camera, give up after RETRY_SECONDS
            now = time.time()
            if failed_since is None:
                failed_since = now
                print("Camera stopped delivering frames, retrying ...")
            if now - failed_since > RETRY_SECONDS:
                print("Camera did not come back. Stopping.")
                self.stop_event.set()
                break
            self.cap.release()
            self.stop_event.wait(0.5)
            self.cap = cv.VideoCapture(self.camera_index)

        self.cap.release()