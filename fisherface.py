import cv2
import os
import numpy as np
import cv2.typing
from typing import List


class FisherfaceFaceRecognizer:
    cascade_classifier: cv2.CascadeClassifier
    faces: List[cv2.typing.MatLike]
    labels: List[int]
    trained: bool

    def __init__(self) -> None:
        self.cascade_classifier = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.model = cv2.face.FisherFaceRecognizer_create()
        self.faces = []
        self.labels = []
        self.trained = False

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise ValueError(f'The model file was not found in the specified directory: {path}. Please ensure the file exists and the path is correct.')
        self.model.read(path)

    def detect_and_resize_face(self, image: cv2.typing.MatLike) -> (cv2.typing.MatLike | None):
        resized_width, resized_height = (50, 38)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = self.cascade_classifier.detectMultiScale(
            gray_image, scaleFactor=1.01, minNeighbors=4, minSize=(30, 30))

        if len(detected_faces) == 0:
            return None

        x, y, w, h = detected_faces[0]
        face_roi = gray_image[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (resized_width, resized_height))

        return resized_face

    def training_data_setup(self, training_data_path: str) -> None:
        self.faces = []
        self.labels = []

        for dir in os.listdir(training_data_path):
            label = dir.split('s')
            label = int(label[1] if len(label) >= 1 else label[0])

            for pathImage in os.listdir(os.path.join(training_data_path, dir)):
                imagePath = os.path.join(training_data_path, dir, pathImage)
                image = cv2.imread(imagePath)
                detected_face = self.detect_and_resize_face(image)

                if detected_face is not None:
                    self.faces.append(detected_face)
                    self.labels.append(label)

                """DEBUG"""
                # if detected_face is None:
                #     print(f'label: {dir}, image: {pathImage}')

    def train(self) -> None:
        if len(self.faces) == 0 or len(self.labels) == 0:
            raise ValueError('The training dataset setup function has not been called before.')

        self.model.train(self.faces, np.array(self.labels))
        self.model.write('classifierFisherface.xml')
        self.trained = True

    def predict(self, test_image: str) -> (tuple[int, float] | tuple[None, None]):
        if not self.trained:
            raise ValueError('The model has not been trained yet. Please train the model first.')
        
        face = self.detect_and_resize_face(test_image)
        if face is not None:
            label, confidence = self.model.predict(face)
            if confidence < 270:
                return label, confidence
            else:
                return None, None
        else:
            return None, None