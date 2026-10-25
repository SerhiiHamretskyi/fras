import cv2 as cv
import numpy as np

MODEL_SIZE = 128   # the model wants a 128x128 RGB image
EXPANSION = 1.5    # the face box is made 1.5x bigger: the model also needs the surroundings


class SpoofDetector:
    """
    Tells a real face from a photo / screen using MiniFASNetV2-SE (ONNX, run with cv2.dnn).
    real_score(frame, bbox) -> 0..1, higher = more likely a real face.
    """

    def __init__(self, model_path):
        self.net = cv.dnn.readNetFromONNX(str(model_path))

    @staticmethod
    def _crop(frame_bgr, bbox):
        """Square crop around the face, 1.5x bigger than the box, edges padded by reflection."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        side = int(max(x2 - x1, y2 - y1) * EXPANSION)
        left = int((x1 + x2) / 2 - side / 2)
        top = int((y1 + y2) / 2 - side / 2)

        # part of the square that lies inside the frame
        cx1, cy1 = max(0, left), max(0, top)
        cx2, cy2 = min(w, left + side), min(h, top + side)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame_bgr[cy1:cy2, cx1:cx2]

        # add what is missing at the frame edges
        pad_top, pad_left = max(0, -top), max(0, -left)
        pad_bottom, pad_right = max(0, top + side - h), max(0, left + side - w)
        crop = cv.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                 cv.BORDER_REFLECT_101)
        return cv.resize(crop, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv.INTER_AREA if side > MODEL_SIZE else cv.INTER_LANCZOS4)

    def real_score(self, frame_bgr, bbox):
        crop = self._crop(frame_bgr, bbox)
        if crop is None:
            return None
        # BGR -> RGB, 0..255 -> 0..1, HWC -> NCHW (no mean/std subtraction in this model)
        blob = cv.dnn.blobFromImage(crop, 1.0 / 255.0, (MODEL_SIZE, MODEL_SIZE), swapRB=True)
        self.net.setInput(blob)
        real_logit, spoof_logit = self.net.forward()[0]
        return float(1.0 / (1.0 + np.exp(spoof_logit - real_logit)))  # = softmax "real" probability