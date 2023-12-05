import os
import sys

import cv2

import cv2.typing


BASE_PATH: str = None

if getattr(sys, "frozen", False):
    BASE_PATH = os.path.dirname(sys.executable)
elif __file__:
    BASE_PATH = os.path.dirname(__file__)

DEFAULT_MODEL_FILE = os.path.join(BASE_PATH, "classifierFisherface.xml")
DEFAULT_TRAINING_DATA = os.path.join(
    BASE_PATH, "dataset", "AT&T Database of Faces", "training-data"
)


class CascadeClassifier:
    cascade_classifier_frontal_face: cv2.CascadeClassifier

    def __init__(self) -> None:
        self.cascade_classifier_frontal_face = cv2.CascadeClassifier(
            os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")
        )

    def detect_face(
        self, img: cv2.typing.MatLike, size: tuple[int, int]
    ) -> cv2.typing.MatLike | None:
        detected_faces = self.cascade_classifier_frontal_face.detectMultiScale(
            img=img, scaleFactor=1.2, minNeighbors=3, minSize=size
        )

        if len(detected_faces) == 0:
            """DEBUG"""
            return None

        return detected_faces[0]
