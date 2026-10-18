import queue
import time
import cv2 as cv

import config
from core.liveness import BlinkCounter, mean_ear
from core.pipeline import FrasPipeline

ENTER, ESC = 13, 27
BACKSPACE = (8, 127)
FONT = cv.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, RED = (0, 255, 0), (0, 200, 255), (0, 0, 255)  # BGR
TIMES = {1: "once", 2: "twice", 3: "three times"}


def name_char(key):
    """Map a key code to a safe filename character, or None."""
    ch = chr(key)
    if ch.isalnum() or ch in "-_":
        return ch
    if ch == " ":
        return "_"
    return None


def main():
    pipeline = FrasPipeline()
    pipeline.start()

    counter = BlinkCounter(config.BLINK_CLOSE_THRESHOLD, config.BLINK_OPEN_THRESHOLD,
                           config.BLINK_MIN_CLOSED_FRAMES, config.BLINK_MAX_CLOSED_SECONDS)

    last_bbox, last_name, last_frame, last_seen = None, None, None, 0.0
    typing, buffer, pending_frame = False, "", None
    message, message_until = "", 0.0

    challenge_name, challenge_start = None, 0.0  # who is being checked, and since when
    live_name = None                             # who has passed the blink check
    next_auto = 0.0                              # no automatic challenge before this time

    def say(text):
        """Speak a phrase as it is ('!' tells VoiceThread it is not a name)."""
        try:
            pipeline.voice_queue.put_nowait("!" + text)
        except queue.Full:
            pass

    def start_challenge(name):
        nonlocal challenge_name, challenge_start
        challenge_name, challenge_start = name, time.time()
        counter.reset()
        n = config.BLINK_TARGET
        say(f"Please blink {TIMES.get(n, str(n) + ' times')}")

    try:
        while not pipeline.stop_event.is_set():
            # 1. New recognition result?
            try:
                res_frame, bbox, name, _ = pipeline.result_queue.get(timeout=0.03)
                new_result = True
            except queue.Empty:
                new_result = False
            now = time.time()

            if new_result:
                last_bbox, last_name, last_frame, last_seen = bbox, name, res_frame, now

                if typing:
                    pass  # stay quiet while a name is being typed
                elif challenge_name is not None:
                    # Blink challenge running: count blinks of the person being checked only
                    if name == challenge_name and now - challenge_start >= config.BLINK_PROMPT_DELAY:
                        ear = mean_ear(res_frame, bbox)
                        if ear is not None:
                            counter.update(ear, now)
                elif name == "Unknown" or name == live_name:
                    # Unknown faces are announced; known faces only after the blink check
                    try:
                        pipeline.voice_queue.put_nowait(name)
                    except queue.Full:
                        pass
                elif config.AUTO_CHALLENGE and now >= next_auto:
                    start_challenge(name)

            # 2. Face gone: the box disappears quickly, the verification lasts longer
            gone = now - last_seen
            if gone > config.BOX_TTL:
                last_bbox = None
                challenge_name = None
            if gone > getattr(config, "LIVE_TTL", 15.0):
                live_name = None

            # 3. Challenge finished?
            if challenge_name is not None:
                if counter.blinks >= config.BLINK_TARGET:
                    live_name, challenge_name = challenge_name, None  # greeting follows
                elif now - challenge_start > config.BLINK_TIMEOUT:
                    challenge_name = None
                    next_auto = now + config.BLINK_RETRY_SECONDS
                    say("Liveness check failed")
                    message, message_until = "Liveness check failed", now + 3

            if not config.SHOW_WINDOW:
                continue

            # 4. Display
            try:
                frame = pipeline.preview_queue.get_nowait().copy()
            except queue.Empty:
                frame = None

            if frame is not None:
                h = frame.shape[0]
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    if last_name == "Unknown":
                        color = RED
                    elif last_name == live_name:
                        color = GREEN
                    else:
                        color = YELLOW
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv.putText(frame, last_name, (x1, y1 - 10), FONT, 0.8, color, 2)
                    if not typing and challenge_name is None:
                        if last_name == "Unknown":
                            cv.putText(frame, "Unknown - press 's' to enroll",
                                       (x1, y2 + 25), FONT, 0.6, RED, 2)
                        elif last_name != live_name:
                            cv.putText(frame, "Press 'c' to verify you are live",
                                       (x1, y2 + 25), FONT, 0.6, YELLOW, 2)

                if challenge_name is not None:
                    left = config.BLINK_TIMEOUT - (now - challenge_start)
                    cv.putText(frame,
                               f"Blink {config.BLINK_TARGET} times: "
                               f"{counter.blinks}/{config.BLINK_TARGET}  ({max(left, 0):.0f}s)",
                               (10, 60), FONT, 0.7, YELLOW, 2)
                if typing:
                    cv.putText(frame, f"Name: {buffer}_   (Enter = save, Esc = cancel)",
                               (10, h - 20), FONT, 0.6, (255, 255, 255), 2)
                if time.time() < message_until:
                    cv.putText(frame, message, (10, 30), FONT, 0.7, (0, 255, 255), 2)

                cv.imshow("FRAS", frame)

            # 5. Keys
            key = cv.waitKey(1) & 0xFF

            if typing:
                if key == ENTER:
                    name = buffer.strip("_")
                    if not name:
                        message = "Name is empty"
                    elif pipeline.store.enroll(pending_frame, name):
                        message = f"Saved: {name}"
                    else:
                        message = "No face found in snapshot, try again"
                    message_until = time.time() + 3
                    typing, pending_frame = False, None
                elif key == ESC:
                    typing, pending_frame = False, None
                elif key in BACKSPACE:
                    buffer = buffer[:-1]
                elif 32 <= key <= 126:
                    ch = name_char(key)
                    if ch and len(buffer) < 30:
                        buffer += ch
            else:
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if last_bbox is None or last_frame is None:
                        message, message_until = "No face in view", time.time() + 2
                    else:
                        pending_frame = last_frame.copy()  # freeze the snapshot
                        buffer, typing = "", True
                        challenge_name = None
                elif key == ord("c") and challenge_name is None:
                    if last_bbox is not None and last_name not in (None, "Unknown"):
                        start_challenge(last_name)
                    else:
                        message, message_until = "No known face to verify", time.time() + 2
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()