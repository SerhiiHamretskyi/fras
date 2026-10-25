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


def landmarks(frame_bgr, bbox):
    """
    Facial landmarks of the face inside bbox (x1, y1, x2, y2), or None.
    Computed once per result and shared by the blink and head-movement checks.
    """
    x1, y1, x2, y2 = bbox
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    # face_recognition wants (top, right, bottom, left); passing it
    # skips its own face detection, so this step stays fast.
    found = face_recognition.face_landmarks(rgb, face_locations=[(y1, x2, y2, x1)])
    return found[0] if found else None


def ear_from(lm):
    """Average EAR of both eyes."""
    return (_eye_aspect_ratio(lm["left_eye"]) + _eye_aspect_ratio(lm["right_eye"])) / 2.0


def mean_ear(frame_bgr, bbox):
    """EAR for the face in bbox, or None if landmarks can't be found."""
    lm = landmarks(frame_bgr, bbox)
    return None if lm is None else ear_from(lm)


def pose_from(lm):
    """
    Rough head pose from 2D landmarks, as two ratios (no angles, no 3D model):
      yaw   = where the nose tip sits between the two sides of the jaw (0 = left edge of the
              image side, 1 = right edge). About 0.5 when facing the camera.
      pitch = how far down the nose tip is between the eye line and the chin.
              Smaller when looking up, bigger when looking down.
    Returns (yaw, pitch) or None.
    """
    chin = lm["chin"]
    jaw_left, jaw_right = chin[0][0], chin[16][0]
    nose = lm["nose_tip"]
    nose_x = sum(p[0] for p in nose) / len(nose)
    nose_y = sum(p[1] for p in nose) / len(nose)
    eyes = lm["left_eye"] + lm["right_eye"]
    eye_y = sum(p[1] for p in eyes) / len(eyes)
    chin_y = chin[8][1]
    if jaw_right - jaw_left <= 0 or chin_y - eye_y <= 0:
        return None
    return (nose_x - jaw_left) / (jaw_right - jaw_left), (nose_y - eye_y) / (chin_y - eye_y)


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


class HeadMoveChecker:
    """
    Decides whether the head has moved left / right / up / down compared with how it
    was held at the start. The start pose (baseline) is the median of the poses added
    with add_baseline(), so every person is measured against their own neutral pose.

    Directions are from the PERSON's point of view and the camera image is not mirrored:
    turning to your left moves the nose to the right side of the image, so yaw goes up.
    """

    def __init__(self, yaw_delta=0.12, pitch_delta=0.08, hold_results=2):
        self.yaw_delta = yaw_delta
        self.pitch_delta = pitch_delta
        self.hold_results = hold_results
        self.reset()

    def reset(self):
        self._yaws, self._pitches = [], []
        self._held = 0

    def add_baseline(self, yaw, pitch):
        self._yaws.append(yaw)
        self._pitches.append(pitch)

    def start_move(self):
        """Call when a new command is given: the hold counter starts from zero."""
        self._held = 0

    @staticmethod
    def _median(values):
        values = sorted(values)
        return values[len(values) // 2]

    def moved(self, direction, yaw, pitch):
        """True once the head has stayed in `direction` for hold_results results in a row."""
        if len(self._yaws) < 3:
            return False  # no baseline yet
        dyaw = yaw - self._median(self._yaws)
        dpitch = pitch - self._median(self._pitches)
        ok = {
            "left": dyaw >= self.yaw_delta,
            "right": dyaw <= -self.yaw_delta,
            "up": dpitch <= -self.pitch_delta,
            "down": dpitch >= self.pitch_delta,
        }[direction]
        self._held = self._held + 1 if ok else 0
        return self._held >= self.hold_results