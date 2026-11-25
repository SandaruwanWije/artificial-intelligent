"""Data augmentation: new examples made from the ones there already are."""

from random import Random


class Augmentation:
    """Copies every example a few times and nudges each copy's inputs a little. The truths stay the same."""

    def __init__(self, copies: int, nudge: float, random: Random):
        self.copies = copies
        self.nudge = nudge
        self.random = random

    def augmented(self,
                  examples: list[tuple[list[float], list[float]]]) -> list[tuple[list[float], list[float]]]:
        """The examples as they are, then the nudged copies of every one."""
        more = list(examples)
        for c in range(self.copies):
            for e in range(len(examples)):
                inputs, truths = examples[e]
                nudged = []
                for i in range(len(inputs)):
                    # As if the same thing were measured once more: close to the first reading, but not equal.
                    nudged.append(inputs[i] + self.random.gauss(0.0, self.nudge))
                more.append((nudged, list(truths)))
        return more
