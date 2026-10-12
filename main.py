import queue
import time
import cv2 as cv

import config
from core.pipeline import FrasPipeline


def main():
    pipeline = FrasPipeline()
    pipeline.start()

    last_bbox, last_name, last_seen = None, None, 0.0
    poll_timeout = 0.03 if config.SHOW_WINDOW else 0.5

    try:
        while not pipeline.stop_event.is_set():
            # 1. New recognition result?
            try:
                _, bbox, name, _ = pipeline.result_queue.get(timeout=poll_timeout)
                last_bbox, last_name, last_seen = bbox, name, time.time()
                try:
                    pipeline.voice_queue.put_nowait(name)
                except queue.Full:
                    pass
            except queue.Empty:
                pass

            # 2. Forget the box if the face has been gone for a moment
            if time.time() - last_seen > config.BOX_TTL:
                last_bbox = None

            # 3. Display (skipped entirely when headless)
            if config.SHOW_WINDOW:
                try:
                    frame = pipeline.preview_queue.get_nowait().copy()
                except queue.Empty:
                    frame = None

                if frame is not None:
                    if last_bbox is not None:
                        x1, y1, x2, y2 = last_bbox
                        color = (0, 255, 0) if last_name != "Unknown" else (0, 0, 255)
                        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv.putText(frame, last_name, (x1, y1 - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv.imshow("FRAS", frame)

                if cv.waitKey(1) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()