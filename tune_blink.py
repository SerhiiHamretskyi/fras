import queue
import threading
import time
import cv2 as cv

import config
from core.capture import CaptureThread
from core.detector import DetectorThread
from core.liveness import mean_ear, BlinkCounter

TARGET_BLINKS = 2
TIMEOUT = 8.0
FONT = cv.FONT_HERSHEY_SIMPLEX

stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=2)
face_queue = queue.Queue(maxsize=2)
preview_queue = queue.Queue(maxsize=2)

threads = [
    CaptureThread(frame_queue, stop_event, config.CAMERA_INDEX, preview_queue),
    DetectorThread(frame_queue, face_queue, stop_event, config.DETECTOR_PROTOTXT,
                   config.DETECTOR_MODEL, config.DETECTOR_CONFIDENCE),
]
for t in threads:
    t.start()

counter = BlinkCounter()
state, started = "idle", 0.0
last_bbox, last_ear, last_seen = None, None, 0.0
updates, fps_t0, fps = 0, time.time(), 0.0

try:
    while True:
        try:
            det_frame, bbox, _ = face_queue.get_nowait()
            ear = mean_ear(det_frame, bbox)
            if ear is not None:
                last_bbox, last_ear, last_seen = bbox, ear, time.time()
                updates += 1
                if state == "running":
                    counter.update(ear, time.time())
        except queue.Empty:
            pass

        if time.time() - last_seen > 0.5:
            last_bbox = last_ear = None

        if state == "running":
            if counter.blinks >= TARGET_BLINKS:
                state = "passed"
            elif time.time() - started > TIMEOUT:
                state = "failed"

        if time.time() - fps_t0 >= 1.0:
            fps, updates, fps_t0 = updates / (time.time() - fps_t0), 0, time.time()

        try:
            frame = preview_queue.get_nowait().copy()
        except queue.Empty:
            frame = None

        if frame is not None:
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ear_txt = f"{last_ear:.2f}" if last_ear is not None else "--"
            cv.putText(frame, f"EAR {ear_txt}   blinks {counter.blinks}   {fps:.0f} fps",
                       (10, 25), FONT, 0.6, (255, 255, 255), 2)
            status = {
                "idle": "Press 'c' to start",
                "running": f"Blink {TARGET_BLINKS} times! {TIMEOUT - (time.time() - started):.0f}s",
                "passed": "LIVE - check passed",
                "failed": "FAILED - press 'c' to retry",
            }[state]
            cv.putText(frame, status, (10, 55), FONT, 0.7, (0, 255, 255), 2)
            cv.imshow("liveness test", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            counter.reset()
            state, started = "running", time.time()
finally:
    stop_event.set()
    cv.destroyAllWindows()