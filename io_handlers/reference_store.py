import os
import face_recognition


class ReferenceStore:
    """
    Loads reference photos from a folder structure (one subfolder per
    person), computes a face encoding for each photo, and keeps them
    in memory for comparison against live face crops.
    """

    def __init__(self, reference_dir: str):
        self.reference_dir = reference_dir
        self.known_encodings = []   # list of 128-d vectors
        self.known_names = []       # parallel list of names
        self._load()

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

                # Upsampling helps catch faces that are smaller/farther
                # from camera in reference photos — default (0) can miss them.
                face_locations = face_recognition.face_locations(
                    image, number_of_times_to_upsample=2
                )

                if len(face_locations) == 0:
                    print(f"Warning: no face found in {filepath}, skipping")
                    continue

                encodings = face_recognition.face_encodings(
                    image, known_face_locations=face_locations
                )

                if len(encodings) == 0:
                    print(f"Warning: could not encode face in {filepath}, skipping")
                    continue

                self.known_encodings.append(encodings[0])
                self.known_names.append(person_name)

        print(f"Loaded {len(self.known_encodings)} reference encodings "
              f"for {len(set(self.known_names))} people")