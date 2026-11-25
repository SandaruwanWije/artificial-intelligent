"""Training: round after round of forward pass, slopes, and one step downhill for every number."""

from neural_network.backpropagation import Backpropagation
from neural_network.forward_pass import ForwardPass
from neural_network.gradient_descent import GradientDescent
from neural_network.network import Network


class Trainer:
    """Finds every slope with backpropagation, then moves every weight and bias with gradient descent."""

    def __init__(self, backpropagation: Backpropagation, gradient_descent: GradientDescent):
        self.backpropagation = backpropagation
        self.gradient_descent = gradient_descent

    def step(self, network: Network, inputs: list[float], truths: list[float]) -> None:
        slopes = self.backpropagation.slopes(network, inputs, truths)
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            for n in range(len(layer.neurons)):
                neuron = layer.neurons[n]
                neuron_slopes = slopes[layer_number][n]
                for i in range(len(neuron.weights)):
                    neuron.weights[i] = self.gradient_descent.step(neuron.weights[i], neuron_slopes.weights[i])
                neuron.bias = self.gradient_descent.step(neuron.bias, neuron_slopes.bias)

    def loss(self, network: Network, examples: list[tuple[list[float], list[float]]]) -> float:
        """The average loss over the examples."""
        total = 0.0
        for inputs, truths in examples:
            answers = ForwardPass(network, inputs).answer
            total = total + self.backpropagation.loss.total(answers, truths)
        return total / len(examples)

    def train(self, network: Network, examples: list[tuple[list[float], list[float]]],
              rounds: int) -> list[float]:
        """Steps once for every example in every round. Returns the loss before training and after each round."""
        losses = [self.loss(network, examples)]
        for _ in range(rounds):
            for inputs, truths in examples:
                self.step(network, inputs, truths)
            losses.append(self.loss(network, examples))
        return losses
