"""The learning rate range test: one short run that shows which learning rates make the loss fall."""

import numpy as np

from neural_network.arrays import ArrayBackpropagation, ArrayNetwork
from neural_network.batches import Batches, Shuffle
from neural_network.tuning.schedule import GrowingRate, ScheduledTrainer


class RateRangeTest:
    """Trains for a few hundred steps while the learning rate grows, and keeps the loss at every rate."""

    def __init__(self, lowest: float, highest: float, steps: int):
        self.schedule = GrowingRate(lowest, highest, steps)
        self.steps = steps
        self.rates: list[float] = []
        self.losses: list[float] = []

    def run(self, network: ArrayNetwork, backpropagation: ArrayBackpropagation,
            inputs: np.ndarray, truths: np.ndarray, batch_size: int, shuffle: Shuffle) -> None:
        """One step for each batch, each at a higher rate. It changes the network, so hand it a spare one."""
        trainer = ScheduledTrainer(network, backpropagation, self.schedule)
        row_numbers = list(range(inputs.shape[0]))
        batches = Batches(batch_size).cut(shuffle.shuffled(row_numbers))
        for step in range(self.steps):
            rows = batches[step]
            # The loss on this batch, before the step at this rate changes the weights.
            self.losses.append(trainer.loss(inputs[rows], truths[rows]))
            self.rates.append(self.schedule.rate(step))
            trainer.step(inputs[rows], truths[rows])

    def smoothed(self) -> list[float]:
        """The losses with their jitter averaged away: each is 9 parts the last average and 1 part the new loss."""
        averages = []
        average = self.losses[0]
        for step in range(len(self.losses)):
            average = 0.9 * average + 0.1 * self.losses[step]
            averages.append(average)
        return averages

    def suggested_rate(self) -> float:
        """A tenth of the rate at which the smoothed loss was lowest: well inside the rates where the loss falls."""
        smoothed = self.smoothed()
        lowest = 0
        for step in range(len(smoothed)):
            if smoothed[step] < smoothed[lowest]:
                lowest = step
        return self.rates[lowest] / 10
