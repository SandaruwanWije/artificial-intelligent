"""The MNIST digits: small pictures of handwritten digits, each with the digit it shows."""

import os
from urllib.request import urlretrieve

import numpy as np


class DigitPictures:
    """Pictures of digits as arrays: one row of 784 inputs and one row of 10 truths for each picture."""

    def __init__(self, pictures: np.ndarray, labels: np.ndarray):
        examples = pictures.shape[0]
        # A picture is 28 rows of 28 pixels. Laid end to end, its rows make one row of 784 numbers.
        flat = pictures.reshape(examples, 784)
        # A pixel is a whole number from 0 (paper) to 255 (ink). Divided by 255, it lies between 0 and 1.
        self.inputs = flat / 255
        # Row 3 of this grid is a 1 in place 3 and a 0 in the nine other places: the truth for a 3.
        one_row_for_each_digit = np.eye(10)
        self.truths = one_row_for_each_digit[labels]
        self.labels = labels


class MnistFile:
    """The MNIST file, cut into three piles: 50,000 pictures to train on, 10,000 to validate and 10,000 to test."""

    ADDRESS = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    def __init__(self, path: str):
        self.path = path

    def download(self) -> None:
        """Fetches the file, about 11 MB, unless it is already there."""
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            urlretrieve(self.ADDRESS, self.path)

    def training(self) -> DigitPictures:
        """What the network learns from."""
        file = np.load(self.path)
        return DigitPictures(file["x_train"][:50000], file["y_train"][:50000])

    def validation(self) -> DigitPictures:
        """What the settings are chosen by: the last 10,000 of the file's training pictures."""
        file = np.load(self.path)
        return DigitPictures(file["x_train"][50000:], file["y_train"][50000:])

    def test(self) -> DigitPictures:
        """Kept away until every choice is made, then used once."""
        file = np.load(self.path)
        return DigitPictures(file["x_test"], file["y_test"])
