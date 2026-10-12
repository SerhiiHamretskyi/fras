import threading
import queue
import time
import pyttsx3


class VoiceThread(threading.Thread):
    def __init__(self, voice_queue: queue.Queue, stop_event: threading.Event,
                 cooldown_seconds: float = 10.0):
        super().__init__(daemon=True)
        self.voice_queue = voice_queue
        self.stop_event = stop_event
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken = {}  # name -> timestamp

    def _message_for(self, name: str) -> str:
        if name == "Unknown":
            return "Unknown person detected"
        return f"Welcome, {name}"

    def run(self):
        engine = pyttsx3.init()

        while not self.stop_event.is_set():
            try:
                name = self.voice_queue.get(timeout=1)
            except queue.Empty:
                continue

            now = time.time()
            last = self._last_spoken.get(name, 0)
            if now - last < self.cooldown_seconds:
                continue

            self._last_spoken[name] = now
            engine.say(self._message_for(name))
            engine.runAndWait() 