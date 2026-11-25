"""Layer norm: one example's numbers are standardized over the neurons of the layer."""

from neural_network.normalization.normalization import Normalization


class LayerNorm(Normalization):
    """Standardizes its inputs over the layer. Then every neuron scales, shifts and bends its own number."""

    def inputs_of(self, n: int, inputs: list[float]) -> list[float]:
        """Neuron number n reads one number: its own input, standardized with the rest of the layer."""
        return [self.standardized(inputs)[n]]

    def outputs(self, inputs: list[float]) -> list[float]:
        outputs = []
        for n in range(len(self.neurons)):
            outputs.append(self.neurons[n].output(self.inputs_of(n, inputs)))
        return outputs

    def blame_on_inputs(self, inputs: list[float], blame_at_totals: list[float]) -> list[float]:
        """Takes the blame back through each neuron's scale, then back through the standardizing."""
        standardized = self.standardized(inputs)
        # Back through the scale: the blame on each standardized number.
        on_standardized = []
        for n in range(len(self.neurons)):
            scale = self.neurons[n].weights[0]
            on_standardized.append(blame_at_totals[n] * scale)
        # Moving one input also moves the mean and the spread, and those move every standardized number.
        through_mean = self.mean(on_standardized)
        pulls = []
        for n in range(len(self.neurons)):
            pulls.append(on_standardized[n] * standardized[n])
        through_spread = self.mean(pulls)
        spread = self.spread(inputs)
        blame = []
        for i in range(len(inputs)):
            direct = on_standardized[i]
            blame.append((direct - through_mean - standardized[i] * through_spread) / spread)
        return blame
