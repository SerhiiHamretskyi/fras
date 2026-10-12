import os
import queue
import threading
import cv2 as cv
from core.capture import CaptureThread
from core.detector import DetectorThread

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(BASE_DIR, "reference_faces")

frame_queue = queue.Queue(maxsize=2)
face_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

capture_thread = CaptureThread(frame_queue, stop_event, camera_index=0)
detector_thread = DetectorThread(
    frame_queue, face_queue, stop_event,
    prototxt_path=os.path.join(BASE_DIR, "models", "deploy.prototxt"),
    model_path=os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
)

capture_thread.start()
detector_thread.start()

last_bbox = None
last_full_frame = None


def save_reference_photo(frame, name: str):
    person_dir = os.path.join(REFERENCE_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = [f for f in os.listdir(person_dir) if f.startswith("photo")]
    next_index = len(existing) + 1
    filepath = os.path.join(person_dir, f"photo{next_index}.jpg")

    cv.imwrite(filepath, frame)
    print(f"Saved {filepath}")


while True:
    try:
        frame = frame_queue.get(timeout=1)
    except queue.Empty:
        frame = None

    try:
        full_frame, bbox, _ = face_queue.get_nowait()
        last_bbox = bbox
        last_full_frame = full_frame
    except queue.Empty:
        pass

    if frame is not None:
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, "Press 's' to save, 'q' to quit", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow('frame', frame)

    key = cv.waitKey(1)

    if key == ord('s'):
        if last_bbox is None or last_full_frame is None:
            print("No face currently detected — can't save.")
        else:
            name = input("Enter name for this face: ").strip()
            if name:
                save_reference_photo(last_full_frame, name)
            else:
                print("Empty name, skipped.")

    elif key == ord('q'):
        stop_event.set()
        break

capture_thread.join()
detector_thread.join()
cv.destroyAllWindows()