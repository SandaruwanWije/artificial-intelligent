"""The perceptron: a neuron with the step as its activation, and the rule that teaches it."""

from neural_network.activations import STEP
from neural_network.neuron import Neuron


class Perceptron(Neuron):
    """A neuron that answers 1 when its total is above 0, and learns from its mistakes."""

    def __init__(self, weights: list[float], bias: float, learning_rate: float = 0.5):
        super().__init__(weights, bias, activation=STEP)
        self.learning_rate = learning_rate

    def guess(self, inputs: list[float]) -> int:
        if self.output(inputs) == 1.0:
            return 1
        return 0

    def learn(self, inputs: list[float], answer: int) -> int:
        """Moves each weight toward the right answer, and returns the error: 1, 0 or -1."""
        error = answer - self.guess(inputs)
        nudge = self.learning_rate * error
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + nudge * inputs[i]
        self.bias = self.bias + nudge
        return error

    def train(self, examples: list[tuple[list[float], int]], max_passes: int = 8) -> list[int]:
        """Goes round the examples until a pass has no mistakes. Returns the mistakes in each pass."""
        mistakes_per_pass = []
        for _ in range(max_passes):
            mistakes = 0
            for inputs, answer in examples:
                error = self.learn(inputs, answer)
                if error != 0:
                    mistakes = mistakes + 1
            mistakes_per_pass.append(mistakes)
            if mistakes == 0:
                break
        return mistakes_per_pass
