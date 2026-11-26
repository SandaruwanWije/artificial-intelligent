"""Training on batches of arrays: a new order each epoch, then one step for every small batch."""

import numpy as np

from neural_network.arrays import ArrayTrainer
from neural_network.batches import Batches, Shuffle
from neural_network.training import Losses


class BatchTrainer(ArrayTrainer):
    """An ArrayTrainer that steps on small batches of rows, instead of on every example in each step."""

    def epoch(self, inputs: np.ndarray, truths: np.ndarray, batch_size: int, shuffle: Shuffle) -> None:
        """One trip through the examples: a new order, then one step for every batch."""
        examples = inputs.shape[0]
        row_numbers = list(range(examples))
        # A new order before every epoch, so no batch holds the same examples twice in a row.
        mixed = shuffle.shuffled(row_numbers)
        batches = Batches(batch_size).cut(mixed)
        for b in range(len(batches)):
            rows = batches[b]
            # A list of row numbers picks those rows out of an array: the batch, as two small arrays.
            self.step(inputs[rows], truths[rows])

    def train_and_validate(self, inputs: np.ndarray, truths: np.ndarray,
                           validation_inputs: np.ndarray, validation_truths: np.ndarray,
                           epochs: int, batch_size: int, shuffle: Shuffle) -> Losses:
        """Steps on the training set only. After each epoch it measures the loss on both sets."""
        losses = Losses([self.loss(inputs, truths)], [self.loss(validation_inputs, validation_truths)])
        for epoch in range(epochs):
            self.epoch(inputs, truths, batch_size, shuffle)
            losses.training.append(self.loss(inputs, truths))
            losses.validation.append(self.loss(validation_inputs, validation_truths))
        return losses
