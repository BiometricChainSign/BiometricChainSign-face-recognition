import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2.typing
import numpy as np
from typing import List

from fisherface import FisherfaceFaceRecognizer


class TestData:
    image: cv2.typing.MatLike
    label: int
    path: str

    def __init__(self, image: cv2.typing.MatLike, label: int, path: str = None) -> None:
        self.image = image
        self.label = label
        self.path = path


class TestOutput:
    expected_value: int
    result_value: int
    image: cv2.typing.MatLike
    face_matched: bool

    def __init__(self, expected_value: int, result_value: int, image: cv2.typing.MatLike, face_matched: bool) -> None:
        self.expected_value = expected_value
        self.result_value = result_value
        self.image = image
        self.face_matched = face_matched


class TestSuite:
    recognizer: FisherfaceFaceRecognizer
    test_data: List[TestData]
    true_labels: List[int | None]
    predicted_labels: List[int | None]
    test_output: List[TestOutput]

    def __init__(self) -> None:
        self.recognizer = FisherfaceFaceRecognizer()
        self.test_data = []
        self.true_labels = []
        self.predicted_labels = []
        self.test_output = []

    def plot_gallery(self, n_row=3, n_col=4):
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01,
                            right=0.99, top=0.90, hspace=0.35)
        i = 0
        for test in self.test_output[0: 12]:
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(test.image)
            plt.title(
                f'Classe prevista: {test.result_value}\nClasse verdadeira: {test.expected_value}', size=12)
            plt.xticks(())
            plt.yticks(())
            i += 1

    def __setup_test_data(self, test_data_path: str) -> None:
        self.test_data = []

        for dir in os.listdir(test_data_path):
            label = None

            if (dir[0] != 's'):
                label = 0
            else:
                label = int(dir.split('s')[1])

            for pathImage in os.listdir(os.path.join(test_data_path, dir)):
                imagePath = os.path.join(test_data_path, dir, pathImage)
                image = cv2.imread(imagePath)
                self.test_data.append(
                    TestData(image=image, label=label, path=imagePath))

        self.test_data.sort(key=lambda x: x.label)

    def run_test(self, training_data_path: str, test_data_path: str) -> List[bool]:
        self.__setup_test_data(test_data_path)
        self.recognizer.setup_training_data(training_data_path)
        self.recognizer.train()
        self.recognizer.load_model('./classifierFisherface.xml')
        self.true_labels: List[int | None] = []
        self.predicted_labels: List[int | None] = []

        for test in self.test_data:
            label, confidence = self.recognizer.predict(test_image=test.image)

            label = label if label is not None else 0

            """DEBUG"""
            if test.label != label:
                print(
                    f'expected label: {test.label}, predicted label: {label}, confidence: {confidence}, path: {test.path}')

            self.true_labels.append(test.label)
            self.predicted_labels.append(label)
            self.test_output.append(TestOutput(expected_value=test.label,
                                               result_value=label, image=test.image, face_matched=test.label == label))

        self.test_output.sort(key=lambda x: x.expected_value)
        self.plot_gallery()
        return list(map(lambda x: x.face_matched, self.test_output))


if __name__ == '__main__':
    test_suite = TestSuite()
    test_result = test_suite.run_test(training_data_path='dataset/AT&T Database of Faces/training-data',
                                      test_data_path='dataset/AT&T Database of Faces/test-data')

    print(classification_report(test_suite.true_labels, test_suite.predicted_labels))

    cm = confusion_matrix(test_suite.true_labels, test_suite.predicted_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.arange(start=1, stop=len(test_result) + 1))
    disp.plot()

    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    mng = plt.get_current_fig_manager()
    mng.set_window_title('BiometricChainSign Face Recognition')
    plt.show()
