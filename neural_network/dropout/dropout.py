"""Dropout: in training, a random part of the hidden neurons is switched off at every pass."""

from random import Random


class Dropout:
    """Switches off a share of the neurons at random, and makes the ones left on count for more."""

    def __init__(self, share: float, random: Random):
        self.share = share
        self.random = random

    def switch(self) -> float:
        """What one neuron's output is multiplied by in this pass. 0 means it is switched off."""
        if self.random.random() < self.share:
            return 0.0
        # The neurons left on are made bigger, so a layer's outputs add up to about as much as with all of them on.
        return 1 / (1 - self.share)


NO_DROPOUT = Dropout(0.0, Random(0))
