"""A confusion grid: for every true digit, how often each digit was read."""

import numpy as np


class Confusion:
    """Counts each pair of the true digit and the digit read. counts[4][9] is how many 4s were read as a 9."""

    def __init__(self, labels: np.ndarray, read: np.ndarray):
        self.counts = np.zeros((10, 10), dtype=int)
        for i in range(len(labels)):
            self.counts[labels[i], read[i]] = self.counts[labels[i], read[i]] + 1

    def right(self) -> int:
        """The pictures read as what they show: the counts where the true digit and the digit read are the same."""
        total = 0
        for digit in range(10):
            total = total + int(self.counts[digit, digit])
        return total

    def worst_mix_up(self) -> tuple[int, int, int]:
        """The wrong pair that happened most: the true digit, the digit read, and how many times."""
        worst = (0, 0, 0)
        for true_digit in range(10):
            for digit_read in range(10):
                count = int(self.counts[true_digit, digit_read])
                if true_digit != digit_read and count > worst[2]:
                    worst = (true_digit, digit_read, count)
        return worst
