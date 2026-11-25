"""Early stopping: end the training when the validation loss stops getting better, and keep the best network."""

from dataclasses import dataclass

from neural_network.batches import Shuffle
from neural_network.network import Network
from neural_network.training import Losses, Trainer


@dataclass
class NeuronNumbers:
    """One neuron's weights and bias, as they were at one moment."""

    weights: list[float]
    bias: float


class EarlyStopping:
    """Trains until the validation loss stops improving, then puts back the network from its best epoch."""

    def __init__(self, trainer: Trainer, patience: int):
        self.trainer = trainer
        self.patience = patience
        self.best_epoch = 0

    def train(self, training: list[tuple[list[float], list[float]]],
              validation: list[tuple[list[float], list[float]]],
              epochs: int, batch_size: int, shuffle: Shuffle) -> Losses:
        """Trains for at most that many epochs. Returns both losses up to the epoch where it stopped."""
        losses = Losses([self.trainer.loss(training)], [self.trainer.loss(validation)])
        best_loss = losses.validation[0]
        best_numbers = self.numbers_of(self.trainer.network)
        self.best_epoch = 0
        for epoch in range(1, epochs + 1):
            self.trainer.epoch(training, batch_size, shuffle)
            losses.training.append(self.trainer.loss(training))
            validation_loss = self.trainer.loss(validation)
            losses.validation.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_numbers = self.numbers_of(self.trainer.network)
                self.best_epoch = epoch
            # Too many epochs with no new best: more training only helps the training score now.
            if epoch - self.best_epoch >= self.patience:
                break
        self.put_back(self.trainer.network, best_numbers)
        return losses

    def numbers_of(self, network: Network) -> list[list[NeuronNumbers]]:
        """A copy of every weight and bias, so later steps cannot change it."""
        numbers = []
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            layer_numbers = []
            for n in range(len(layer.neurons)):
                neuron = layer.neurons[n]
                layer_numbers.append(NeuronNumbers(list(neuron.weights), neuron.bias))
            numbers.append(layer_numbers)
        return numbers

    def put_back(self, network: Network, numbers: list[list[NeuronNumbers]]) -> None:
        """Sets every weight and bias of the network to the kept copy."""
        for layer_number in range(len(network.layers)):
            layer = network.layers[layer_number]
            for n in range(len(layer.neurons)):
                kept = numbers[layer_number][n]
                layer.neurons[n].weights = list(kept.weights)
                layer.neurons[n].bias = kept.bias
