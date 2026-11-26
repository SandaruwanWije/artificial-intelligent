"""Searching settings: which values to try when several dials must be set together."""

import math
from abc import ABC, abstractmethod
from random import Random


class Dial:
    """One setting to tune, somewhere between its lowest and highest value."""

    def __init__(self, name: str, lowest: float, highest: float):
        self.name = name
        self.lowest = lowest
        self.highest = highest

    def evenly(self, count: int) -> list[float]:
        """Values from the lowest to the highest, each the same number of times the one before: 0.001, 0.01, 0.1."""
        times = math.exp(math.log(self.highest / self.lowest) / (count - 1))
        values = []
        value = self.lowest
        for i in range(count):
            values.append(value)
            value = value * times
        return values

    def anywhere(self, random: Random) -> float:
        """One random value. It is as likely to fall between 0.001 and 0.01 as between 0.1 and 1."""
        how_far = random.random()
        return self.lowest * math.exp(how_far * math.log(self.highest / self.lowest))


class Search(ABC):
    """Chooses the trials to run. A trial is one value for every dial, kept under the dial's name."""

    def __init__(self, dials: list[Dial]):
        self.dials = list(dials)

    @abstractmethod
    def trials(self) -> list[dict[str, float]]: ...


class GridSearch(Search):
    """A few values for each dial, and every way of putting them together."""

    def __init__(self, dials: list[Dial], values_per_dial: int):
        super().__init__(dials)
        self.values_per_dial = values_per_dial

    def trials(self) -> list[dict[str, float]]:
        trials = [{}]
        for dial in self.dials:
            # Every trial so far is copied once for each value of this dial.
            longer = []
            for trial in trials:
                for value in dial.evenly(self.values_per_dial):
                    with_this_dial = dict(trial)
                    with_this_dial[dial.name] = value
                    longer.append(with_this_dial)
            trials = longer
        return trials


class RandomSearch(Search):
    """Every trial draws a fresh random value for every dial, so no value is tried twice."""

    def __init__(self, dials: list[Dial], count: int, random: Random):
        super().__init__(dials)
        self.count = count
        self.random = random

    def trials(self) -> list[dict[str, float]]:
        trials = []
        for t in range(self.count):
            trial = {}
            for dial in self.dials:
                trial[dial.name] = dial.anywhere(self.random)
            trials.append(trial)
        return trials
