"""Measuring a slope without its formula, by nudging a weight and watching the loss."""

from collections.abc import Callable

from .bowl import Bowl, Gradient, Weights

NUDGE = 1e-4


def feel_slope(loss: Callable[[float], float], weight: float, nudge: float = NUDGE) -> float:
    loss_change = loss(weight + nudge) - loss(weight - nudge)
    return loss_change / (2 * nudge)


def feel_gradient(bowl: Bowl, weights: Weights, nudge: float = NUDGE) -> Gradient:
    def loss_moving_only_w1(w1: float) -> float:
        return bowl.loss(Weights(w1, weights.w2))

    def loss_moving_only_w2(w2: float) -> float:
        return bowl.loss(Weights(weights.w1, w2))

    return Gradient(
        slope_for_w1=feel_slope(loss_moving_only_w1, weights.w1, nudge),
        slope_for_w2=feel_slope(loss_moving_only_w2, weights.w2, nudge),
    )
