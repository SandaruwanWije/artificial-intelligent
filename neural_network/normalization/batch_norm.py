"""Batch norm: one neuron's numbers are standardized over the examples of the batch."""

from neural_network.activations import Activation
from neural_network.normalization.normalization import Normalization


class BatchNorm(Normalization):
    """Standardizes each neuron's input over the batch. In use it standardizes with what it kept from training."""

    def __init__(self, size: int, activation: Activation, keep: float = 0.9):
        super().__init__(size, activation)
        self.keep = keep
        self.means = []  # each neuron's mean over the batch it is training on
        self.spreads = []
        self.kept_means = []  # running averages of those, for when the network is used
        self.kept_spreads = []
        for n in range(size):
            self.means.append(0.0)
            self.spreads.append(1.0)
            self.kept_means.append(0.0)
            self.kept_spreads.append(1.0)
        self.through_mean = []  # each neuron's two averages of the blame over the batch
        self.through_spread = []

    def see_inputs(self, batch_inputs: list[list[float]]) -> None:
        """Measures each neuron's mean and spread over the batch, and adds a little of them to the kept ones."""
        for n in range(len(self.neurons)):
            column = []
            for e in range(len(batch_inputs)):
                column.append(batch_inputs[e][n])
            self.means[n] = self.mean(column)
            self.spreads[n] = self.spread(column)
            self.kept_means[n] = self.keep * self.kept_means[n] + (1 - self.keep) * self.means[n]
            self.kept_spreads[n] = self.keep * self.kept_spreads[n] + (1 - self.keep) * self.spreads[n]

    def inputs_of(self, n: int, inputs: list[float]) -> list[float]:
        """Neuron number n reads one number: its own input, standardized over the batch."""
        return [(inputs[n] - self.means[n]) / self.spreads[n]]

    def outputs(self, inputs: list[float]) -> list[float]:
        """In use there is no batch, so each input is standardized with the kept mean and spread."""
        outputs = []
        for n in range(len(self.neurons)):
            standardized = (inputs[n] - self.kept_means[n]) / self.kept_spreads[n]
            outputs.append(self.neurons[n].output([standardized]))
        return outputs

    def see_blame(self, batch_inputs: list[list[float]], batch_blame_at_totals: list[list[float]]) -> None:
        """Measures, for each neuron over the batch, the two averages its blame needs on the way back."""
        self.through_mean = []
        self.through_spread = []
        for n in range(len(self.neurons)):
            scale = self.neurons[n].weights[0]
            on_standardized = []
            pulls = []
            for e in range(len(batch_inputs)):
                standardized = self.inputs_of(n, batch_inputs[e])[0]
                direct = batch_blame_at_totals[e][n] * scale
                on_standardized.append(direct)
                pulls.append(direct * standardized)
            self.through_mean.append(self.mean(on_standardized))
            self.through_spread.append(self.mean(pulls))

    def blame_on_inputs(self, inputs: list[float], blame_at_totals: list[float]) -> list[float]:
        """Takes the blame back through each neuron's scale, then back through the standardizing."""
        blame = []
        for n in range(len(self.neurons)):
            standardized = self.inputs_of(n, inputs)[0]
            direct = blame_at_totals[n] * self.neurons[n].weights[0]
            # Moving this input also moved the neuron's mean and spread, and those moved every example's number.
            through = self.through_mean[n] + standardized * self.through_spread[n]
            blame.append((direct - through) / self.spreads[n])
        return blame
