"""The forward pass: inputs go through the network one layer at a time, and every step is kept."""

from dataclasses import dataclass

from neural_network.dropout import NO_DROPOUT, Dropout
from neural_network.network import Layer, Network


@dataclass
class LayerStep:
    """What one layer got, what each of its neurons added up to, what came out, and which neurons were on."""

    inputs: list[float]
    totals: list[float]
    outputs: list[float]
    switches: list[float]


class ForwardPass:
    """Runs inputs through a network and keeps each layer's step, so they can be looked at later."""

    def __init__(self, network: Network, inputs: list[float], dropout: Dropout = NO_DROPOUT):
        self.steps: list[LayerStep] = []
        self.dropout = dropout
        self.values = inputs  # what the next layer will get
        last = len(network.layers) - 1
        for layer_number in range(len(network.layers)):
            self.through_layer(network.layers[layer_number], layer_number == last)

    def through_layer(self, layer: Layer, is_last: bool) -> None:
        """Takes the pass through one more layer and keeps that layer's step."""
        totals = []
        outputs = []
        switches = []
        for n in range(len(layer.neurons)):
            neuron = layer.neurons[n]
            total = neuron.total(layer.inputs_of(n, self.values))
            # Dropout may switch a hidden neuron off for this pass. The last layer always stays on.
            switch = 1.0
            if not is_last:
                switch = self.dropout.switch()
            totals.append(total)
            outputs.append(neuron.activation(total) * switch)
            switches.append(switch)
        self.steps.append(LayerStep(self.values, totals, outputs, switches))
        self.values = outputs

    @property
    def answer(self) -> list[float]:
        last_step = self.steps[len(self.steps) - 1]
        return last_step.outputs


NO_LAYERS = Network([])


class BatchForwardPass:
    """Takes a whole batch through a network one layer at a time, so a layer can look at every example first."""

    def __init__(self, network: Network, batch_inputs: list[list[float]], dropout: Dropout = NO_DROPOUT):
        # Every example gets a pass of its own, which has gone through no layers yet.
        self.passes: list[ForwardPass] = []
        for e in range(len(batch_inputs)):
            self.passes.append(ForwardPass(NO_LAYERS, batch_inputs[e], dropout))
        last = len(network.layers) - 1
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            arriving = []
            for e in range(len(self.passes)):
                arriving.append(self.passes[e].values)
            layer.see_inputs(arriving)
            for e in range(len(self.passes)):
                self.passes[e].through_layer(layer, layer_number == last)
