import os
import cv2

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from typing import List
import cv2.typing

from fisherface import FisherfaceFaceRecognizer


class TestData:
    test_img: cv2.typing.MatLike
    true_img: cv2.typing.MatLike
    label: int
    path: str

    def __init__(
        self,
        label: int,
        test_img: cv2.typing.MatLike,
        true_img: cv2.typing.MatLike,
        path: str = None,
    ) -> None:
        self.label = label
        self.test_img = test_img
        self.true_img = true_img
        self.path = path


class TestOutput:
    expected_value: int
    result_value: int
    test_img: cv2.typing.MatLike
    true_img: cv2.typing.MatLike
    face_matched: bool

    def __init__(
        self,
        expected_value: int,
        result_value: int,
        test_img: cv2.typing.MatLike,
        true_img: cv2.typing.MatLike,
        face_matched: bool,
    ) -> None:
        self.expected_value = expected_value
        self.result_value = result_value
        self.test_img = test_img
        self.true_img = true_img
        self.face_matched = face_matched


class TestSuite:
    recognizer: FisherfaceFaceRecognizer
    test_data: List[TestData]
    true_labels: List[int | None]
    predicted_labels: List[int | None]
    test_output: List[TestOutput]
    tested: bool

    def __init__(self) -> None:
        self.recognizer = FisherfaceFaceRecognizer()
        self.test_data = []
        self.true_labels = []
        self.predicted_labels = []
        self.test_output = []
        self.tested = False

    def face_not_found(self, image: cv2.typing.MatLike, text: str):
        if image is None:
            return None

        blurred_background = cv2.GaussianBlur(image, (15, 15), 0)

        text_color = (255, 0, 0)

        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        font_thickness = 1
        

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2

        image_with_text = np.copy(blurred_background)
        cv2.putText(image_with_text, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return image_with_text

    def plot_gallery(self, n_row=3, n_col=4):
        if not self.tested:
            raise ValueError(
                "The test must be executed before plotting the image.")

        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01,
                            right=0.99, top=0.90, hspace=0.35)
        i = 0
        NUM_IMAGES_TO_DISPLAY = 12
        

        white_separator = np.full(
            (self.test_output[0].test_img.shape[0], 10, 3), 255, dtype=np.uint8
        )

        for test in self.test_output[0:NUM_IMAGES_TO_DISPLAY]:
            plt.subplot(n_row, n_col, i + 1)

            plt.imshow(
                np.concatenate(
                    (test.test_img, white_separator, (test.true_img if test.face_matched else self.face_not_found(test.true_img, "ERRO"))), axis=1
                )
            )

            plt.title(
                f"imagem de teste               imagem resultante\nclasse: {test.expected_value}                             classe: {test.result_value}",
                size=10,
            )
            plt.xticks(())
            plt.yticks(())
            i += 1
        plt.get_current_fig_manager().set_window_title(
            "BiometricChainSign Face Recognition Test"
        )

    def __test_data_setup(self, test_data_path: str, training_data_path: str) -> None:
        self.test_data = []

        for dir in os.listdir(test_data_path):
            if dir.startswith("."):
                continue

            label = None

            if dir[0] != "s":
                label = 0
            else:
                label = int(dir.split("s")[1])

            for pathImg in os.listdir(os.path.join(test_data_path, dir)):
                if pathImg.startswith("."):
                    continue
                imgPath = os.path.join(test_data_path, dir, pathImg)
                test_img = cv2.imread(imgPath)
                true_img = cv2.imread(
                    os.path.join(
                        training_data_path,
                        dir,
                        os.listdir(os.path.join(training_data_path, dir))[0],
                    )
                )

                self.test_data.append(
                    TestData(
                        label=label,
                        test_img=test_img,
                        true_img=true_img,
                        path=imgPath,
                    )
                )

        self.test_data.sort(key=lambda x: x.label)

    def run_test(self, test_data_path: str, training_data_path: str) -> List[bool]:
        self.__test_data_setup(
            test_data_path=test_data_path, training_data_path=training_data_path
        )

        self.recognizer.training_data_setup(training_data_path)
        self.recognizer.train()
        self.recognizer.load_model()

        self.true_labels: List[int | None] = []
        self.predicted_labels: List[int | None] = []

        for test in self.test_data:
            label, confidence = self.recognizer.predict(test_img=test.test_img)

            label = label if label is not None else 0

            """DEBUG"""
            # if test.label != label:
            #     print(
            #         f'expected label: {test.label}, predicted label: {label}, confidence: {confidence}, path: {test.path}')

            self.true_labels.append(test.label)
            self.predicted_labels.append(label)
            self.test_output.append(
                TestOutput(
                    expected_value=test.label,
                    result_value=label,
                    test_img=test.test_img,
                    true_img=test.true_img,
                    face_matched=test.label == label,
                )
            )

        self.test_output.sort(key=lambda x: x.expected_value)
        self.tested = True
        return list(map(lambda x: x.face_matched, self.test_output))


if __name__ == "__main__":
    test_suite = TestSuite()
    test_result = test_suite.run_test(
        test_data_path="dataset/AT&T Database of Faces/test-data",
        training_data_path="dataset/AT&T Database of Faces/training-data",
    )

    test_suite.plot_gallery()

    print(classification_report(test_suite.true_labels, test_suite.predicted_labels))

    cm = confusion_matrix(test_suite.true_labels, test_suite.predicted_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        # display_labels=np.arange(start=1, stop=len(test_result) + 1),
    )
    disp.plot()

    plt.title("Matriz de confusão")
    plt.xticks(rotation=45)
    plt.xlabel(xlabel="Rótulos previstos")
    plt.ylabel(ylabel="Rótulos verdadeiros")

    mng = plt.get_current_fig_manager()
    mng.set_window_title("BiometricChainSign")
    plt.show()
