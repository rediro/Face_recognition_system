import pickle
from pathlib import Path

import face_recognition

DEFAULT_ENCODINGS_PATH = Path("encodings1.pkl")

# Create directories if they don't already exist



def encode_known_faces(
    model: str = "cnn", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []
    for filepath in Path("/home/jetson/CameraCalibration/faces").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


encode_known_faces()