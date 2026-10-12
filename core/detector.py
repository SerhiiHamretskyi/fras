
import threading
import queue
import cv2 as cv


class DetectorThread(threading.Thread):
    """
    Runs face detection on every frame from frame_queue.
    Pushes (frame, bbox, face_crop) to face_queue only when a face is found.
    """

    def __init__(self, frame_queue: queue.Queue, face_queue: queue.Queue,
                 stop_event: threading.Event,
                 prototxt_path: str, model_path: str,
                 confidence_threshold: float = 0.5):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.face_queue = face_queue
        self.stop_event = stop_event
        self.confidence_threshold = confidence_threshold
        self.net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]

            # Build input blob: resize to 300x300, mean-subtraction values
            # are fixed constants this specific model was trained with.
            blob = cv.dnn.blobFromImage(
                cv.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()

            # detections shape: [1, 1, N, 7] — loop over N candidate detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < self.confidence_threshold:
                    continue

                # Box coords are normalized [0,1] — scale back to frame size
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")

                # Clip to frame bounds (model can predict slightly outside)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                bbox = (x1, y1, x2, y2)

                if self.face_queue.full():
                    try:
                        self.face_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.face_queue.put((frame, bbox, face_crop))