"""Backpropagation: the slope of the loss for every weight and bias, in one sweep backwards."""

from dataclasses import dataclass

from neural_network.forward_pass import ForwardPass, LayerStep
from neural_network.loss import Loss
from neural_network.network import Layer, Network


@dataclass
class NeuronSlopes:
    """How the loss moves when each of one neuron's weights, or its bias, moves a little."""

    weights: list[float]
    bias: float


class Backpropagation:
    """Starts at the loss and walks back through the layers, sharing out the blame."""

    def __init__(self, loss: Loss):
        self.loss = loss

    def slopes(self, network: Network, inputs: list[float], truths: list[float]) -> list[list[NeuronSlopes]]:
        """One list per layer, one NeuronSlopes per neuron, in the same order as the network."""
        forward = ForwardPass(network, inputs)
        # The blame starts at the answers: how the loss moves when each answer moves.
        blame = []
        for i in range(len(truths)):
            blame.append(self.loss.slope(forward.answer[i], truths[i]))
        # Then it walks back, from the last layer to the first.
        all_slopes = []
        layer_number = len(network.layers) - 1
        while layer_number >= 0:
            layer = network.layers[layer_number]
            step = forward.steps[layer_number]
            layer_slopes, blame = self.back_through_layer(layer, step, blame)
            all_slopes.insert(0, layer_slopes)
            layer_number = layer_number - 1
        return all_slopes

    def back_through_layer(self, layer: Layer, step: LayerStep,
                           blame: list[float]) -> tuple[list[NeuronSlopes], list[float]]:
        """Takes the blame on a layer's outputs. Returns its neurons' slopes and the blame on its inputs."""
        neuron_slopes = []
        blame_at_totals = []
        for n in range(len(layer.neurons)):
            neuron = layer.neurons[n]
            # Back through the bend: times the activation's slope at this neuron's total.
            at_total = blame[n] * neuron.activation.slope(step.totals[n])
            blame_at_totals.append(at_total)
            # A weight's slope is the blame at the total times the input it multiplied.
            weight_slopes = []
            for x in step.inputs:
                weight_slopes.append(at_total * x)
            # The bias is added straight to the total, so its slope is the blame at the total.
            neuron_slopes.append(NeuronSlopes(weight_slopes, at_total))
        # Each input gets blame back along every weight it went through.
        blame_on_inputs = []
        for i in range(len(step.inputs)):
            shared = 0.0
            for n in range(len(layer.neurons)):
                shared = shared + blame_at_totals[n] * layer.neurons[n].weights[i]
            blame_on_inputs.append(shared)
        return neuron_slopes, blame_on_inputs
