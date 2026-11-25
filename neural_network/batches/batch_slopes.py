"""Batch slopes: the slope of every weight and bias, averaged over a batch of examples."""

from neural_network.backpropagation import Backpropagation, NeuronSlopes
from neural_network.network import Network


class BatchSlopes:
    """Finds each example's slopes with backpropagation, then averages them."""

    def __init__(self, backpropagation: Backpropagation):
        self.backpropagation = backpropagation

    def slopes(self, network: Network,
               batch: list[tuple[list[float], list[float]]]) -> list[list[NeuronSlopes]]:
        """One list per layer, one NeuronSlopes per neuron: each slope averaged over the batch."""
        # First, every example's own slopes.
        each_example = self.backpropagation.slopes_for_each(network, batch)
        # Then the average, one neuron at a time.
        averages = []
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            layer_averages = []
            for n in range(len(layer.neurons)):
                one_neuron = []
                for e in range(len(batch)):
                    one_neuron.append(each_example[e][layer_number][n])
                layer_averages.append(self.average(one_neuron))
            averages.append(layer_averages)
        return averages

    def average(self, slopes: list[NeuronSlopes]) -> NeuronSlopes:
        """One neuron's slopes from every example go in. Their average comes out."""
        count = len(slopes)
        weights = []
        for i in range(len(slopes[0].weights)):
            total = 0.0
            for e in range(count):
                total = total + slopes[e].weights[i]
            weights.append(total / count)
        total = 0.0
        for e in range(count):
            total = total + slopes[e].bias
        return NeuronSlopes(weights, total / count)
