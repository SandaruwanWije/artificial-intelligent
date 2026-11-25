"""Backpropagation: the slope of the loss for every weight and bias, in one sweep backwards."""

from dataclasses import dataclass

from neural_network.dropout import NO_DROPOUT, Dropout
from neural_network.forward_pass import BatchForwardPass, LayerStep
from neural_network.loss import Loss
from neural_network.network import Layer, Network


@dataclass
class NeuronSlopes:
    """How the loss moves when each of one neuron's weights, or its bias, moves a little."""

    weights: list[float]
    bias: float


class Backpropagation:
    """Starts at the loss and walks back through the layers, sharing out the blame."""

    def __init__(self, loss: Loss, dropout: Dropout = NO_DROPOUT):
        self.loss = loss
        self.dropout = dropout

    def slopes(self, network: Network, inputs: list[float], truths: list[float]) -> list[list[NeuronSlopes]]:
        """One list per layer, one NeuronSlopes per neuron, in the same order as the network."""
        # One example is a batch of one.
        return self.slopes_for_each(network, [(inputs, truths)])[0]

    def slopes_for_each(self, network: Network,
                        batch: list[tuple[list[float], list[float]]]) -> list[list[list[NeuronSlopes]]]:
        """The slopes for every example of a batch. All of them go forward, then back, one layer at a time."""
        batch_inputs = []
        for e in range(len(batch)):
            batch_inputs.append(batch[e][0])
        forward = BatchForwardPass(network, batch_inputs, self.dropout)
        # The blame starts at the answers: how the loss moves when each answer moves.
        blames = []
        all_slopes = []
        for e in range(len(batch)):
            answer = forward.passes[e].answer
            truths = batch[e][1]
            blame = []
            for i in range(len(truths)):
                blame.append(self.loss.slope(answer[i], truths[i]))
            blames.append(blame)
            all_slopes.append([])
        # Then it walks back, from the last layer to the first.
        layer_number = len(network.layers) - 1
        while layer_number >= 0:
            layer = network.layers[layer_number]
            layer_inputs = []
            at_totals = []
            for e in range(len(batch)):
                step = forward.passes[e].steps[layer_number]
                layer_slopes, blame_at_totals = self.back_to_totals(layer, step, blames[e])
                all_slopes[e].insert(0, layer_slopes)
                layer_inputs.append(step.inputs)
                at_totals.append(blame_at_totals)
            # The layer may look at every example's blame. Then it shares each example's blame back to its inputs.
            layer.see_blame(layer_inputs, at_totals)
            for e in range(len(batch)):
                blames[e] = layer.blame_on_inputs(layer_inputs[e], at_totals[e])
            layer_number = layer_number - 1
        return all_slopes

    def back_to_totals(self, layer: Layer, step: LayerStep,
                       blame: list[float]) -> tuple[list[NeuronSlopes], list[float]]:
        """Takes the blame on a layer's outputs. Returns its neurons' slopes and the blame at their totals."""
        neuron_slopes = []
        blame_at_totals = []
        for n in range(len(layer.neurons)):
            neuron = layer.neurons[n]
            # Back through the switch and the bend. A neuron that dropout switched off passes no blame back.
            through_switch = blame[n] * step.switches[n]
            at_total = through_switch * neuron.activation.slope(step.totals[n])
            blame_at_totals.append(at_total)
            # A weight's slope is the blame at the total times the input it multiplied.
            weight_slopes = []
            for x in layer.inputs_of(n, step.inputs):
                weight_slopes.append(at_total * x)
            # The bias is added straight to the total, so its slope is the blame at the total.
            neuron_slopes.append(NeuronSlopes(weight_slopes, at_total))
        return neuron_slopes, blame_at_totals
