from pytest import approx

from neural_network.activations import GELU, NO_BEND, RELU, SIGMOID, STEP, TANH


def trace(show, curve, totals):
    """Shows the curve's value and slope at each total."""
    for z in totals:
        show("total", z, "-> value", curve(z), "  slope", curve.slope(z))


def test_step_jumps_at_zero_and_has_no_slope(show):
    trace(show, STEP, [-1.0, -0.01, 0.01, 1.0])
    assert (STEP(-1), STEP(1)) == (0, 1)
    assert STEP.slope(1) == 0


def test_sigmoid_is_half_at_zero_and_steepest_there(show):
    trace(show, SIGMOID, [-3.0, 0.0, 3.0])
    assert SIGMOID(0) == 0.5
    assert SIGMOID.slope(0) == 0.25
    assert SIGMOID.slope(3) == approx(0.045, abs=0.001)


def test_tanh_goes_from_minus_1_to_1(show):
    trace(show, TANH, [-5.0, 0.0, 5.0])
    assert TANH(0) == 0
    assert TANH.slope(0) == 1
    assert TANH(5) == approx(1, abs=0.001)


def test_relu_keeps_only_the_positive_part(show):
    trace(show, RELU, [-3.0, 0.0, 2.0])
    assert [RELU(z) for z in (-3, 0, 2)] == [0, 0, 2]
    assert [RELU.slope(z) for z in (-3, 3)] == [0, 1]


def test_gelu_dips_below_zero(show):
    trace(show, GELU, [-3.0, 0.0, 2.0])
    lowest, at = min((GELU(z / 100), z / 100) for z in range(-300, 1))
    show("lowest value", lowest, "at total", at)
    assert lowest == approx(-0.17, abs=0.005)
    assert GELU.slope(0) == 0.5


def test_no_bend_changes_nothing(show):
    trace(show, NO_BEND, [-7.0, 2.5])
    assert NO_BEND(2.5) == 2.5
    assert NO_BEND.slope(-7) == 1


def test_every_slope_matches_a_tiny_nudge(show):
    nudge = 1e-6
    for curve in (SIGMOID, TANH, RELU, GELU, NO_BEND):
        for z in (-2.0, -0.5, 0.7, 2.0):
            felt = (curve(z + nudge) - curve(z - nudge)) / (2 * nudge)
            assert curve.slope(z) == approx(felt, abs=1e-6)
        z = 0.7
        felt = (curve(z + nudge) - curve(z - nudge)) / (2 * nudge)
        show(type(curve).__name__, "at total 0.7: slope", curve.slope(z), "  nudge test", felt)
