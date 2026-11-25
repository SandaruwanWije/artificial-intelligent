from pytest import approx

from neural_network.gradient_descent import GradientDescent


def valley_loss(weight: float) -> float:
    """One weight and its loss: lowest (0.3) at weight 1."""
    return 0.35 * (weight - 1) ** 2 + 0.3


def valley_slope(weight: float) -> float:
    return 0.7 * (weight - 1)


def test_one_step_goes_against_the_slope(show):
    descent = GradientDescent(learning_rate=0.8)
    weight = -2.2
    slope = valley_slope(weight)
    show("weight", weight, "  slope", slope, "  the slope is negative, so downhill is to the right")
    show("new weight = weight - 0.8 x slope =", descent.step(weight, slope))
    assert slope == approx(-2.24)
    assert descent.step(weight, slope) == approx(-0.408)


def test_steps_walk_down_to_the_lowest_point(show):
    descent = GradientDescent(learning_rate=0.8)
    weight = -2.2
    show("start   weight", weight, "loss", valley_loss(weight))
    for step in range(1, 31):
        weight = descent.step(weight, valley_slope(weight))
        if step in (1, 2, 3, 5, 10, 30):
            show("step", step, "  weight", weight, "loss", valley_loss(weight))
    assert weight == approx(1, abs=0.001)
    assert valley_loss(weight) == approx(0.3)


def test_too_big_a_step_climbs_out(show):
    descent = GradientDescent(learning_rate=3.0)
    weight = -2.2
    show("start   weight", weight, "loss", valley_loss(weight))
    for step in range(1, 6):
        weight = descent.step(weight, valley_slope(weight))
        show("step", step, "  weight", weight, "loss", valley_loss(weight))
    assert valley_loss(weight) > valley_loss(-2.2)
