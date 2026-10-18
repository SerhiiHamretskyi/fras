import os
import cv2 as cv
import face_recognition


class ReferenceStore:
    """
    Holds encodings of known faces. Loads them from reference_faces/
    at startup and can enroll new faces while the program is running.
    """

    def __init__(self, reference_dir: str):
        self.reference_dir = reference_dir
        self.known_encodings = []
        self.known_names = []
        self._load()

    @staticmethod
    def _encode(image_rgb):
        """Return the encoding of the first face found, or None."""
        locations = face_recognition.face_locations(
            image_rgb, number_of_times_to_upsample=2
        )
        if not locations:
            return None
        encodings = face_recognition.face_encodings(
            image_rgb, known_face_locations=locations
        )
        return encodings[0] if encodings else None

    def _add(self, name, encoding):
        # Names first, then encodings: the recognizer thread reads these
        # lists without a lock, so any index valid for encodings must
        # also be valid for names.
        self.known_names.append(name)
        self.known_encodings.append(encoding)

    def _load(self):
        if not os.path.isdir(self.reference_dir):
            print(f"Reference directory {self.reference_dir} does not exist yet.")
            return

        for person_name in os.listdir(self.reference_dir):
            person_dir = os.path.join(self.reference_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                filepath = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encoding = self._encode(image)

                if encoding is None:
                    print(f"Warning: no face found in {filepath}, skipping")
                    continue
                self._add(person_name, encoding)

        print(f"Loaded {len(self.known_encodings)} reference encodings "
              f"for {len(set(self.known_names))} people")

    def enroll(self, frame_bgr, name: str):
        """
        Save a snapshot under reference_faces/<name>/ and add the face
        to memory right away. Returns the saved path, or None if no
        face was found (nothing is saved in that case).
        """
        encoding = self._encode(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
        if encoding is None:
            return None

        person_dir = os.path.join(self.reference_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        index = len([f for f in os.listdir(person_dir) if f.startswith("photo")]) + 1
        path = os.path.join(person_dir, f"photo{index}.jpg")
        cv.imwrite(path, frame_bgr)

        self._add(name, encoding)
        return path