"""Training: epoch after epoch of batches, slopes, and one step downhill for every number."""

from dataclasses import dataclass

from neural_network.backpropagation import Backpropagation
from neural_network.batches import Batches, BatchSlopes, Shuffle
from neural_network.gradient_clipping import NO_LIMIT, GradientClipping
from neural_network.network import Network
from neural_network.optimizers import NeuronOptimizers, Optimizer
from neural_network.weight_decay import NO_DECAY, WeightDecay


@dataclass
class Losses:
    """The loss before training and after each epoch, on the training set and on the validation set."""

    training: list[float]
    validation: list[float]


class Trainer:
    """Trains one network: finds every slope with backpropagation, then moves each number with its own optimizer."""

    def __init__(self, network: Network, backpropagation: Backpropagation, optimizer: Optimizer,
                 clipping: GradientClipping = NO_LIMIT, weight_decay: WeightDecay = NO_DECAY):
        self.network = network
        self.backpropagation = backpropagation
        self.batch_slopes = BatchSlopes(backpropagation)
        self.optimizers = self.one_for_each_number(optimizer)
        self.clipping = clipping
        self.weight_decay = weight_decay

    def one_for_each_number(self, optimizer: Optimizer) -> list[list[NeuronOptimizers]]:
        """A fresh optimizer for every weight and bias, so each one has a memory of its own."""
        all_optimizers = []
        for layer_number in range(len(self.network.layers)):
            layer = self.network.layers[layer_number]
            layer_optimizers = []
            for n in range(len(layer.neurons)):
                for_weights = []
                for i in range(len(layer.neurons[n].weights)):
                    for_weights.append(optimizer.fresh())
                layer_optimizers.append(NeuronOptimizers(for_weights, optimizer.fresh()))
            all_optimizers.append(layer_optimizers)
        return all_optimizers

    def step(self, batch: list[tuple[list[float], list[float]]]) -> None:
        """One step for every weight and bias, from its average slope over the batch."""
        slopes = self.batch_slopes.slopes(self.network, batch)
        slopes = self.weight_decay.added(self.network, slopes)
        slopes = self.clipping.clipped(slopes)
        for layer_number in range(len(self.network.layers)):
            layer = self.network.layers[layer_number]
            for n in range(len(layer.neurons)):
                neuron = layer.neurons[n]
                neuron_slopes = slopes[layer_number][n]
                own = self.optimizers[layer_number][n]
                for i in range(len(neuron.weights)):
                    neuron.weights[i] = own.weights[i].step(neuron.weights[i], neuron_slopes.weights[i])
                neuron.bias = own.bias.step(neuron.bias, neuron_slopes.bias)

    def loss(self, examples: list[tuple[list[float], list[float]]]) -> float:
        """The average loss over the examples."""
        total = 0.0
        for inputs, truths in examples:
            answers = self.network.answer(inputs)
            total = total + self.backpropagation.loss.total(answers, truths)
        return total / len(examples)

    def epoch(self, examples: list[tuple[list[float], list[float]]], batch_size: int, shuffle: Shuffle) -> None:
        """One trip through the examples: a new order, then one step for every batch."""
        # A new order before every epoch, so no batch holds the same examples twice in a row.
        mixed = shuffle.shuffled(examples)
        batches = Batches(batch_size).cut(mixed)
        for b in range(len(batches)):
            self.step(batches[b])

    def train(self, examples: list[tuple[list[float], list[float]]],
              epochs: int, batch_size: int, shuffle: Shuffle) -> list[float]:
        """Trains for some epochs. Returns the loss before training and after each epoch."""
        losses = [self.loss(examples)]
        for epoch in range(epochs):
            self.epoch(examples, batch_size, shuffle)
            losses.append(self.loss(examples))
        return losses

    def train_and_validate(self, training: list[tuple[list[float], list[float]]],
                           validation: list[tuple[list[float], list[float]]],
                           epochs: int, batch_size: int, shuffle: Shuffle) -> Losses:
        """Steps on the training set only. After each epoch it measures the loss on both sets."""
        losses = Losses([self.loss(training)], [self.loss(validation)])
        for epoch in range(epochs):
            self.epoch(training, batch_size, shuffle)
            losses.training.append(self.loss(training))
            losses.validation.append(self.loss(validation))
        return losses
