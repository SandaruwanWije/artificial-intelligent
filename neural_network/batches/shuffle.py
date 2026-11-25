"""Shuffle: the same examples, in a new random order."""

from random import Random


class Shuffle:
    """Mixes examples into a random order, so every batch is a new mix."""

    def __init__(self, random: Random):
        self.random = random

    def shuffled(self, examples: list) -> list:
        """A mixed copy. The list it was given keeps its order."""
        mixed = list(examples)
        # Walk back from the last place. Each place swaps with a random place at or before it.
        place = len(mixed) - 1
        while place > 0:
            other = self.random.randrange(place + 1)
            kept = mixed[place]
            mixed[place] = mixed[other]
            mixed[other] = kept
            place = place - 1
        return mixed
