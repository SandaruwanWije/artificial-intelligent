"""How far the network's answer falls short of the right one."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    guess: float
    right_answer: float

    @property
    def short_by(self) -> float:
        return self.right_answer - self.guess
