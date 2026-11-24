"""The numbers the article quotes, checked against the code."""

import math

import pytest

from ai_02_activations.__main__ import A_HUNDRED_LAYERS, LAYER_1, LAYER_2, RING, TRAINER
from ai_02_activations.activations import GELU, NoBend, ReLU, Sigmoid, Step, Tanh
from ai_02_activations.bump import BUMP, N1, N2, N3, XS, goal
from ai_02_activations.layers import stack
from ai_02_activations.network import TinyNetwork


@pytest.fixture(scope="module")
def runs():
    return {activation.name: TRAINER.train(TinyNetwork(activation, 4, seed=19), RING)
            for activation in (NoBend(), ReLU(), Tanh())}


def test_step_has_a_slope_of_0_and_none_at_the_jump():
    step = Step()
    assert (step.value(-1), step.slope(-1)) == (0, 0)
    assert (step.value(1), step.slope(1)) == (1, 0)
    assert math.isnan(step.slope(0))


def test_sigmoid_at_minus_2_0_and_2():
    sigmoid = Sigmoid()
    assert round(sigmoid.value(-2), 2) == 0.12 and round(sigmoid.slope(-2), 3) == 0.105
    assert sigmoid.value(0) == 0.5 and sigmoid.slope(0) == 0.25
    assert round(sigmoid.value(2), 2) == 0.88 and round(sigmoid.slope(2), 3) == 0.105


def test_tanh_slope_reaches_1_in_the_middle():
    assert Tanh().slope(0) == 1


def test_relu_slope_is_exactly_0_or_1():
    assert ReLU().slope(-0.5) == 0 and ReLU().slope(0.5) == 1


def test_gelu_dips_to_minus_0_17_near_minus_0_75():
    z, lowest = GELU().lowest([z / 100 for z in range(-400, 1)])
    assert (z, round(lowest, 2)) == (-0.75, -0.17)


def test_slopes_at_3():
    assert round(Sigmoid().slope(3), 3) == 0.045
    assert round(Tanh().slope(3), 2) == 0.01
    assert ReLU().slope(3) == 1 and round(GELU().slope(3)) == 1


def test_the_goal_is_the_bump():
    assert [goal(x) for x in XS] == [0, 0, 1, 0, 0]


def test_layer_1_gives_1_5_and_layer_2_gives_2():
    assert LAYER_1(2) == 1.5
    assert LAYER_2(LAYER_1(2)) == 2


def test_layers_with_no_bend_stack_into_x():
    assert (LAYER_1.then(LAYER_2).w, LAYER_1.then(LAYER_2).b) == (1, 0)
    together = stack(A_HUNDRED_LAYERS)
    assert len(A_HUNDRED_LAYERS) == 100 and (together.w, together.b) == (1, 0)


def test_relu_neurons_by_hand():
    assert [N1.z(x) for x in XS] == [-1, 0, 1, 2, 3]
    assert [N1.gives(x) for x in XS] == [0, 0, 1, 2, 3]
    assert [N2.gives(x) for x in XS] == [0, 0, 0, 1, 2]
    assert [N3.gives(x) for x in XS] == [0, 0, 0, 0, 1]
    assert [n.corner for n in (N1, N2, N3)] == [0, 1, 2]


def test_the_final_neuron_draws_the_bump():
    assert BUMP.weights == (1, -2, 1)
    assert [BUMP.y(x) for x in XS] == [0, 0, 1, 0, 0]


def test_the_ring_has_200_points():
    assert len(RING) == 200


def test_first_50_steps_match_the_chart(runs):
    steps = (0, 10, 20, 30, 40, 50)
    assert [runs["ReLU"].percent_right(s) for s in steps] == [33, 73, 78, 83, 86, 85]
    assert [runs["Tanh"].percent_right(s) for s in steps] == [33, 63, 59, 60, 65, 78]
    assert [runs["No bend"].percent_right(s) for s in steps] == [34, 58, 60, 56, 58, 56]


def test_no_bend_stalls_at_57_percent(runs):
    assert runs["No bend"].percent_right() == 57


def test_relu_gets_all_200_right_after_150_steps(runs):
    assert runs["ReLU"].right_after_each_step[150] == 200
    assert runs["ReLU"].percent_right() == 100


def test_tanh_closes_the_ring_after_100_steps(runs):
    assert runs["Tanh"].percent_right(100) == 100
    assert runs["Tanh"].percent_right() == 100
