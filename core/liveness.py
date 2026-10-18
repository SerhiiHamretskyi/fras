import math
import cv2 as cv
import face_recognition


def _eye_aspect_ratio(eye):
    """eye = 6 (x, y) points in dlib order: p1 corner, p2/p3 upper lid,
    p4 other corner, p5/p6 lower lid."""
    p1, p2, p3, p4, p5, p6 = eye
    vertical = math.dist(p2, p6) + math.dist(p3, p5)
    horizontal = 2.0 * math.dist(p1, p4)
    return vertical / horizontal


def mean_ear(frame_bgr, bbox):
    """
    Average EAR of both eyes for the face inside bbox (x1, y1, x2, y2).
    Returns None if landmarks can't be found.
    """
    x1, y1, x2, y2 = bbox
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    # face_recognition wants (top, right, bottom, left); passing it
    # skips its own face detection, so this step stays fast.
    landmarks = face_recognition.face_landmarks(
        rgb, face_locations=[(y1, x2, y2, x1)]
    )
    if not landmarks:
        return None
    left = _eye_aspect_ratio(landmarks[0]["left_eye"])
    right = _eye_aspect_ratio(landmarks[0]["right_eye"])
    return (left + right) / 2.0


class BlinkCounter:
    """
    Counts blinks from a stream of EAR values.
    A blink = EAR below close_threshold for at least min_closed_frames,
    then back above open_threshold, within max_closed_seconds.
    Two different thresholds (hysteresis) stop flicker at the boundary.
    """

    def __init__(self, close_threshold=0.21, open_threshold=0.25,
                 min_closed_frames=2, max_closed_seconds=0.5):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_closed_frames = min_closed_frames
        self.max_closed_seconds = max_closed_seconds
        self.reset()

    def reset(self):
        self.blinks = 0
        self._closed = False
        self._closed_frames = 0
        self._closed_since = 0.0

    def update(self, ear, now):
        if not self._closed:
            if ear < self.close_threshold:
                self._closed = True
                self._closed_frames = 1
                self._closed_since = now
        else:
            if ear < self.open_threshold:
                self._closed_frames += 1
            else:
                duration = now - self._closed_since
                if (self._closed_frames >= self.min_closed_frames
                        and duration <= self.max_closed_seconds):
                    self.blinks += 1
                self._closed = False
        return self.blinks