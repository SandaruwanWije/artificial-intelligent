"""The numbers the article quotes, checked against the code."""

from pytest import approx

from ai_06_gradient_descent.__main__ import (BOWL, BOWL_DESCENT, BOWL_START, LEARNING_RATES,
                                             STEPS_ON_THE_CHART, VALLEY, VALLEY_DESCENT,
                                             VALLEY_START)
from ai_06_gradient_descent.bowl import Gradient
from ai_06_gradient_descent.descent import GradientDescent
from ai_06_gradient_descent.feel import feel_gradient, feel_slope


def valley_steps():
    return VALLEY_DESCENT.down_the_valley(VALLEY, VALLEY_START, steps=5).steps


def bowl_losses(name):
    rate = LEARNING_RATES[name]
    return GradientDescent(rate).across_the_bowl(BOWL, BOWL_START, STEPS_ON_THE_CHART).losses


def test_the_valley_starts_at_minus_2_2_with_a_loss_of_3_88():
    assert VALLEY.loss(VALLEY_START) == approx(3.88, abs=0.005)


def test_the_slope_is_minus_2_24_then_minus_0_99():
    steps = valley_steps()
    assert steps[0].slope == approx(-2.24, abs=0.005)
    assert steps[1].weight == approx(-0.41, abs=0.005)
    assert steps[1].slope == approx(-0.99, abs=0.005)


def test_a_nudge_feels_the_same_slope_as_the_formula():
    for weight in (VALLEY_START, valley_steps()[1].weight):
        assert feel_slope(VALLEY.loss, weight) == approx(VALLEY.slope(weight))


def test_the_steps_down_the_valley_match_the_table():
    steps = valley_steps()
    assert [round(step.weight, 2) for step in steps[:4]] == [-2.2, -0.41, 0.38, 0.73]
    assert [round(step.slope, 2) for step in steps[:4]] == [-2.24, -0.99, -0.43, -0.19]


def test_the_steps_shrink_from_1_79_to_0_07():
    assert [round(step.size, 2) for step in valley_steps()] == [1.79, 0.79, 0.35, 0.15, 0.07]


def test_the_bowl_starts_with_a_loss_of_7_55():
    assert BOWL.loss(BOWL_START) == approx(7.55)


def test_the_gradient_at_the_start_is_minus_3_2_and_5_4():
    assert BOWL.gradient(BOWL_START) == Gradient(approx(-3.2), approx(5.4))
    assert feel_gradient(BOWL, BOWL_START) == Gradient(approx(-3.2), approx(5.4))


def test_one_step_at_0_15_moves_both_weights_and_lands_on_3_724():
    descent = BOWL_DESCENT.across_the_bowl(BOWL, BOWL_START, steps=1)
    after = descent.path[1]
    assert (after.w1 - BOWL_START.w1, after.w2 - BOWL_START.w2) == (approx(0.48), approx(-0.81))
    assert descent.losses[1] == approx(3.724, abs=0.0005)


def test_0_03_is_still_at_1_12_after_25_steps():
    assert bowl_losses("too small")[25] == approx(1.117, abs=0.0005)


def test_0_15_is_close_to_0_by_step_15():
    losses = bowl_losses("about right")
    assert losses[15] == approx(0.039, abs=0.0005)
    assert losses[25] == approx(0.002, abs=0.0005)


def test_0_34_falls_to_about_3_5_then_climbs_to_17_27():
    losses = bowl_losses("too big")
    assert min(losses) == approx(3.498, abs=0.0005)
    assert losses[25] == approx(17.269, abs=0.0005)
