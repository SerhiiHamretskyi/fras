# core/recognizer.py

import threading
import queue
import numpy as np
import face_recognition


class RecognizerThread(threading.Thread):
    def __init__(self, face_queue, result_queue, stop_event, reference_store,
                 distance_threshold=0.6):
        super().__init__(daemon=True)
        self.face_queue = face_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.reference_store = reference_store
        self.distance_threshold = distance_threshold

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame, bbox, face_crop = self.face_queue.get(timeout=1)
            except queue.Empty:
                continue

            face_crop = np.ascontiguousarray(face_crop)  # <-- the fix

            h, w = face_crop.shape[:2]
            known_location = [(0, w, h, 0)]

            encodings = face_recognition.face_encodings(
                face_crop, known_face_locations=known_location
            )
            if len(encodings) == 0:
                continue

            encoding = encodings[0]

            if len(self.reference_store.known_encodings) == 0:
                name, distance = "Unknown", None
            else:
                distances = face_recognition.face_distance(
                    self.reference_store.known_encodings, encoding
                )
                best_index = distances.argmin()
                best_distance = distances[best_index]
                name = (self.reference_store.known_names[best_index]
                        if best_distance <= self.distance_threshold else "Unknown")
                distance = best_distance

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put((frame, bbox, name, distance))