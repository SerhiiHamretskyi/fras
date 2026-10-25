"""
Tuning tool for the head-move challenge. Run:  python test_pose.py
Look straight at the camera and press 'b' (sets your neutral pose), then turn your head
left / right / up / down and watch the 'change' numbers. 'q' quits.
Pick HEAD_YAW_DELTA / HEAD_PITCH_DELTA in config.py a bit below the change you get
from a comfortable (not extreme) head movement.
"""
import cv2 as cv
import face_recognition

import config
from core.liveness import landmarks, pose_from

FONT = cv.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv.VideoCapture(config.CAMERA_INDEX)
    base = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera")
            break

        small = cv.resize(frame, None, fx=0.5, fy=0.5)
        boxes = face_recognition.face_locations(cv.cvtColor(small, cv.COLOR_BGR2RGB))
        pose = None
        if boxes:
            top, right, bottom, left = [v * 2 for v in boxes[0]]
            lm = landmarks(frame, (left, top, right, bottom))
            pose = pose_from(lm) if lm else None
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if pose:
            yaw, pitch = pose
            cv.putText(frame, f"yaw {yaw:.2f}  pitch {pitch:.2f}", (10, 30), FONT, 0.7, (0, 255, 255), 2)
            if base:
                cv.putText(frame, f"change: yaw {yaw - base[0]:+.2f}  pitch {pitch - base[1]:+.2f}",
                           (10, 60), FONT, 0.7, (0, 255, 255), 2)
        cv.putText(frame, "b = set neutral pose, q = quit", (10, frame.shape[0] - 15), FONT, 0.6, (255, 255, 255), 2)
        cv.imshow("pose test", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b") and pose:
            base = pose
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()