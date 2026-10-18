import queue
import subprocess
import threading
import time


class VoiceThread(threading.Thread):
    """
    Reads names from a queue and speaks a greeting through espeak-ng.
    Each name has a cooldown so the same person isn't announced every frame.
    An item starting with "!" is a phrase that is spoken as it is.
    """

    def __init__(self, voice_queue: queue.Queue, stop_event: threading.Event,
                 cooldown_seconds: float = 10.0, speed: int = 150):
        super().__init__(daemon=True)
        self.voice_queue = voice_queue
        self.stop_event = stop_event
        self.cooldown_seconds = cooldown_seconds
        self.speed = speed  # words per minute
        self._last_spoken = {}  # name -> timestamp

    @staticmethod
    def _message_for(name: str) -> str:
        if name == "Unknown":
            return "Unknown person detected"
        spoken = name.replace("_", " ").replace("-", " ").title()
        return f"Welcome, {spoken}"

    def _speak(self, text: str):
        try:
            # "--" stops a name starting with "-" from being read as an option
            subprocess.run(
                ["espeak-ng", "-s", str(self.speed), "--", text],
                check=False, timeout=15,
            )
        except FileNotFoundError:
            print("espeak-ng not found: sudo apt install espeak-ng")
        except subprocess.TimeoutExpired:
            print("Voice timed out")

    def run(self):
        while not self.stop_event.is_set():
            try:
                name = self.voice_queue.get(timeout=1)
            except queue.Empty:
                continue

            if name.startswith("!"):  # a phrase to say as it is, no cooldown
                self._speak(name[1:])
                continue

            now = time.time()
            if now - self._last_spoken.get(name, 0) < self.cooldown_seconds:
                continue

            self._last_spoken[name] = now
            self._speak(self._message_for(name))  # blocks this thread only