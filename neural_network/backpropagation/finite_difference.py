"""Finite differences: every slope the slow way, by nudging one number at a time."""

from neural_network.backpropagation.backpropagation import NeuronSlopes
from neural_network.loss import Loss
from neural_network.network import Network


class FiniteDifference:
    """Nudges one weight or bias, runs the network again, and divides the change in loss by the nudge."""

    def __init__(self, loss: Loss, nudge: float = 1e-6):
        self.loss = loss
        self.nudge = nudge

    def loss_of(self, network: Network, inputs: list[float], truths: list[float]) -> float:
        """The loss with the network's numbers as they are now."""
        answers = network.answer(inputs)
        return self.loss.total(answers, truths)

    def slope(self, before: float, after: float) -> float:
        """How far the loss moved, for each unit the number moved."""
        return (after - before) / self.nudge

    def slopes(self, network: Network, inputs: list[float], truths: list[float]) -> list[list[NeuronSlopes]]:
        """One list per layer, one NeuronSlopes per neuron. Needs one more run of the network per number."""
        before = self.loss_of(network, inputs, truths)
        all_slopes = []
        for layer in network.layers:
            layer_slopes = []
            for neuron in layer.neurons:
                weight_slopes = []
                for i in range(len(neuron.weights)):
                    kept = neuron.weights[i]
                    neuron.weights[i] = kept + self.nudge  # only this number moves
                    after = self.loss_of(network, inputs, truths)
                    neuron.weights[i] = kept  # put it back before the next one
                    weight_slopes.append(self.slope(before, after))
                kept = neuron.bias
                neuron.bias = kept + self.nudge
                after = self.loss_of(network, inputs, truths)
                neuron.bias = kept
                layer_slopes.append(NeuronSlopes(weight_slopes, self.slope(before, after)))
            all_slopes.append(layer_slopes)
        return all_slopes
