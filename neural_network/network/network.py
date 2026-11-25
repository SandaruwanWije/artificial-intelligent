"""Layers of neurons, and a network that passes each layer's outputs to the next layer."""

from neural_network.neuron import Neuron


class Layer:
    """Neurons side by side. Every neuron gets the same inputs."""

    def __init__(self, neurons: list[Neuron]):
        self.neurons = list(neurons)

    def outputs(self, inputs: list[float]) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(inputs))
        return outputs

    def inputs_of(self, n: int, inputs: list[float]) -> list[float]:
        """What neuron number n reads. In a plain layer every neuron reads all the inputs."""
        return inputs

    def blame_on_inputs(self, inputs: list[float], blame_at_totals: list[float]) -> list[float]:
        """Takes the blame at each neuron's total. Returns the blame on each of the layer's inputs."""
        blame = []
        for i in range(len(inputs)):
            # Each input gets blame back along every weight it went through.
            shared = 0.0
            for n in range(len(self.neurons)):
                shared = shared + blame_at_totals[n] * self.neurons[n].weights[i]
            blame.append(shared)
        return blame

    def see_inputs(self, batch_inputs: list[list[float]]) -> None:
        """Shown every example's inputs before any example goes through. A plain layer has no use for them."""

    def see_blame(self, batch_inputs: list[list[float]], batch_blame_at_totals: list[list[float]]) -> None:
        """Shown every example's blame at the totals before any of it goes back. A plain layer has no use for it."""


class Network:
    """Layers in a row. Each layer's outputs are the next layer's inputs."""

    def __init__(self, layers: list[Layer]):
        self.layers = list(layers)

    def answer(self, inputs: list[float]) -> list[float]:
        values = inputs
        for layer in self.layers:
            values = layer.outputs(values)
        return values
