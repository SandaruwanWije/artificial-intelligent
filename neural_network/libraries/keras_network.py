"""The same network in Keras: the layers are listed, and one call runs the whole training loop."""

import os

# Keras does its sums with another library underneath. This picks PyTorch, and it must be set before Keras is loaded.
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from neural_network.arrays import ArrayNetwork


class KerasNetwork:
    """Some inputs, a hidden layer of ReLU neurons and one sigmoid answer, as a Keras model."""

    def __init__(self, inputs: int, hidden: int, learning_rate: float):
        self.model = keras.Sequential([
            keras.Input(shape=(inputs,)),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        optimizer = keras.optimizers.SGD(learning_rate)
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy")

    def start_from(self, network: ArrayNetwork) -> None:
        """Copies an array network's weights and biases. Keras keeps one column for each neuron, as the arrays do."""
        numbers = []
        for layer in network.layers:
            numbers.append(layer.weights)
            numbers.append(layer.biases)
        self.model.set_weights(numbers)

    def answer(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs, verbose=0)

    def train(self, inputs: np.ndarray, truths: np.ndarray, epochs: int) -> list[float]:
        """Trains for some epochs, every example in each step. Returns the loss before training and after each epoch."""
        examples = len(inputs)
        # One batch holds every example, in the same order each time, as the arrays train. Nothing is printed.
        history = self.model.fit(inputs, truths, epochs=epochs, batch_size=examples, shuffle=False, verbose=0)
        # Keras measures each epoch's loss before it moves the weights, so the loss after the last step is missing.
        losses = list(history.history["loss"])
        losses.append(float(self.model.evaluate(inputs, truths, batch_size=examples, verbose=0)))
        return losses
