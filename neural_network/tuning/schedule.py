"""Schedules: a learning rate that changes from step to step while a network trains."""

import math
from abc import ABC, abstractmethod

import numpy as np

from neural_network.arrays import ArrayBackpropagation, ArrayNetwork
from neural_network.digits import BatchTrainer
from neural_network.gradient_descent import GradientDescent


class Schedule(ABC):
    """Says what the learning rate is at each step of a run. Steps are counted from 0."""

    @abstractmethod
    def rate(self, step: int) -> float: ...


class Steady(Schedule):
    """One learning rate for the whole run."""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def rate(self, step: int) -> float:
        return self.learning_rate


class WarmUpThenDecay(Schedule):
    """Climbs in a straight line to the peak, then glides down to 0 along half a cosine wave."""

    def __init__(self, peak: float, warm_up_steps: int, total_steps: int):
        self.peak = peak
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps

    def rate(self, step: int) -> float:
        if step < self.warm_up_steps:
            # Warm up: small steps while the weights are still random, a little larger each time.
            return self.peak * (step + 1) / self.warm_up_steps
        # Decay: how far through the rest of the run this step is, from 0 at the peak to 1 at the end.
        done = (step - self.warm_up_steps) / (self.total_steps - self.warm_up_steps)
        # cos goes from 1 down to -1 over half a wave, so this goes from 1 down to 0.
        share_of_the_peak = (1 + math.cos(math.pi * done)) / 2
        return self.peak * share_of_the_peak


class GrowingRate(Schedule):
    """Starts at the lowest rate and is multiplied by the same number at every step, to end at the highest."""

    def __init__(self, lowest: float, highest: float, steps: int):
        self.lowest = lowest
        # Growing by the same share each step covers 0.001 to 0.01 in as many steps as 0.1 to 1.
        self.growth = math.exp(math.log(highest / lowest) / (steps - 1))

    def rate(self, step: int) -> float:
        rate = self.lowest
        for s in range(step):
            rate = rate * self.growth
        return rate


class ScheduledTrainer(BatchTrainer):
    """A BatchTrainer whose learning rate follows a schedule: before each step it asks for that step's rate."""

    def __init__(self, network: ArrayNetwork, backpropagation: ArrayBackpropagation, schedule: Schedule):
        super().__init__(network, backpropagation, GradientDescent(schedule.rate(0)))
        self.schedule = schedule
        self.steps_taken = 0

    def step(self, inputs: np.ndarray, truths: np.ndarray) -> None:
        self.gradient_descent.learning_rate = self.schedule.rate(self.steps_taken)
        super().step(inputs, truths)
        self.steps_taken = self.steps_taken + 1
