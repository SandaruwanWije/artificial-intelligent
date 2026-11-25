"""The forward pass: inputs go through the network one layer at a time, and every step is kept."""

from dataclasses import dataclass

from neural_network.network import Network


@dataclass
class LayerStep:
    """What one layer got, what each of its neurons added up to, and what came out."""

    inputs: list[float]
    totals: list[float]
    outputs: list[float]


class ForwardPass:
    """Runs inputs through a network and keeps each layer's step, so they can be looked at later."""

    def __init__(self, network: Network, inputs: list[float]):
        self.steps: list[LayerStep] = []
        values = inputs
        for layer in network.layers:
            totals = []
            outputs = []
            for neuron in layer.neurons:
                total = neuron.total(values)
                totals.append(total)
                outputs.append(neuron.activation(total))
            self.steps.append(LayerStep(values, totals, outputs))
            values = outputs

    @property
    def answer(self) -> list[float]:
        last_step = self.steps[len(self.steps) - 1]
        return last_step.outputs
