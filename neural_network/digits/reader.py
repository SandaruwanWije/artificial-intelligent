"""A digit reader: the array network with 784 inputs, one hidden layer and ten scores."""

from random import Random

import numpy as np

from neural_network.activations import NO_BEND, RELU
from neural_network.arrays import ARRAY_RELU, ArrayLayer, ArrayNetwork
from neural_network.digits.pictures import DigitPictures
from neural_network.digits.softmax import ARRAY_NO_BEND, Softmax
from neural_network.initialization import He, Xavier


class DigitReader(ArrayNetwork):
    """An ArrayNetwork that reads a digit: 784 pixels in, a layer of ReLU neurons, ten scores out."""

    def __init__(self, hidden_neurons: int, random: Random):
        # Random starting weights, sized by each layer's inputs: He for the ReLU layer, Xavier for the scores.
        hidden = He(random).layer(784, hidden_neurons, RELU)
        scores = Xavier(random).layer(hidden_neurons, 10, NO_BEND)
        super().__init__([ArrayLayer(hidden, ARRAY_RELU), ArrayLayer(scores, ARRAY_NO_BEND)])
        self.softmax = Softmax()

    def chances(self, inputs: np.ndarray) -> np.ndarray:
        """Ten chances for each picture: how sure the network is of each digit."""
        scores = self.answer(inputs)
        return self.softmax.chances(scores)

    def digits(self, inputs: np.ndarray) -> np.ndarray:
        """The digit read from each picture: the place of the largest of its ten scores."""
        scores = self.answer(inputs)
        return scores.argmax(axis=1)

    def right(self, pictures: DigitPictures) -> int:
        """How many of the pictures it reads as the digit they show."""
        read = self.digits(pictures.inputs)
        matches = read == pictures.labels
        return int(matches.sum())
