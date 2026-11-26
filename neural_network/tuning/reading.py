"""Reading loss curves: what a run's training and validation loss say to fix first."""

import math

from neural_network.training import Losses


class CurveReading:
    """Reads the two loss curves of one run. Ask its four questions in order: the first yes is what to fix."""

    def __init__(self, losses: Losses, good_enough: float):
        self.training = losses.training
        self.validation = losses.validation
        # The training loss you would be happy with. Only you can say what is low enough for your problem.
        self.good_enough = good_enough

    def epochs_the_loss_went_up(self) -> int:
        """How many epochs ended with a higher training loss than the epoch before."""
        count = 0
        for epoch in range(1, len(self.training)):
            if self.training[epoch] > self.training[epoch - 1]:
                count = count + 1
        return count

    def rate_is_too_high(self) -> bool:
        """The loss is not a number any more, or it jumps up and down and has not even halved."""
        last = self.training[len(self.training) - 1]
        if math.isnan(last) or math.isinf(last):
            return True
        epochs = len(self.training) - 1
        jumps_about = self.epochs_the_loss_went_up() * 10 >= epochs * 3
        has_halved = last < self.training[0] / 2
        return jumps_about and not has_halved

    def rate_is_too_low(self) -> bool:
        """The loss is still too high, and it was still falling well in the second half of the run."""
        last = self.training[len(self.training) - 1]
        halfway = self.training[len(self.training) // 2]
        still_falling = last < halfway * 0.8
        return last > self.good_enough and still_falling

    def underfits(self) -> bool:
        """The loss is still too high, and it has gone flat: the network is too small to do better."""
        last = self.training[len(self.training) - 1]
        return last > self.good_enough and not self.rate_is_too_low()

    def best_epoch(self) -> int:
        """The epoch with the lowest validation loss."""
        best = 0
        for epoch in range(len(self.validation)):
            if self.validation[epoch] < self.validation[best]:
                best = epoch
        return best

    def overfits(self) -> bool:
        """The validation loss ends more than a tenth above its best: it turned up while training kept falling."""
        last = self.validation[len(self.validation) - 1]
        best = self.validation[self.best_epoch()]
        return last > best * 1.1

    def says(self) -> str:
        """What to fix first, in the order the fixes should be made."""
        if self.rate_is_too_high():
            return "learning rate too high"
        if self.rate_is_too_low():
            return "learning rate too low"
        if self.underfits():
            return "underfitting"
        if self.overfits():
            return "overfitting"
        return "healthy"
