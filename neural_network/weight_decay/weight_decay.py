"""Weight decay: large weights cost something, so every weight is pulled a little toward 0."""

from neural_network.backpropagation import NeuronSlopes
from neural_network.network import Network


class WeightDecay:
    """Adds a price for large weights to the loss, and the slope of that price to every weight's slope."""

    def __init__(self, strength: float):
        self.strength = strength

    def penalty(self, network: Network) -> float:
        """What the weights cost: half the strength times every weight squared, all added up. Biases are free."""
        squares = 0.0
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            for n in range(len(layer.neurons)):
                neuron = layer.neurons[n]
                for i in range(len(neuron.weights)):
                    squares = squares + neuron.weights[i] * neuron.weights[i]
        return 0.5 * self.strength * squares

    def added(self, network: Network, slopes: list[list[NeuronSlopes]]) -> list[list[NeuronSlopes]]:
        """The slopes with the penalty's slope added: the strength times the weight, for every weight."""
        all_slopes = []
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            layer_slopes = []
            for n in range(len(layer.neurons)):
                neuron = layer.neurons[n]
                neuron_slopes = slopes[layer_number][n]
                weights = []
                for i in range(len(neuron.weights)):
                    # A big weight gets a big push back toward 0. A weight at 0 gets none.
                    pull = self.strength * neuron.weights[i]
                    weights.append(neuron_slopes.weights[i] + pull)
                layer_slopes.append(NeuronSlopes(weights, neuron_slopes.bias))
            all_slopes.append(layer_slopes)
        return all_slopes


NO_DECAY = WeightDecay(0.0)
