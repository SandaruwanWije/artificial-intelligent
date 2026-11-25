import math

from pytest import approx

from neural_network.loss import CrossEntropy, SquaredError

SQUARED, CROSS = SquaredError(), CrossEntropy()


def test_squared_error_is_the_miss_times_itself(show):
    for answer in (0.59, 3):
        show("answer", answer, "truth 1  miss", answer - 1, "  cost", SQUARED.cost(answer, 1))
    assert SQUARED.cost(0.59, 1) == approx(0.1681)
    assert SQUARED.cost(3, 1) == 4


def test_twice_the_miss_costs_four_times_as_much(show):
    for answer in (1.2, 1.4):
        show("miss", answer - 1, "-> cost", SQUARED.cost(answer, 1))
    assert SQUARED.cost(1.2, 1) * 4 == approx(SQUARED.cost(1.4, 1))


def test_cross_entropy_is_small_when_sure_and_right_and_huge_when_sure_and_wrong(show):
    for answer, truth in ((0.5, 1), (0.99, 1), (0.01, 1), (0.01, 0)):
        show("answer", answer, "truth", truth, "-> cost", CROSS.cost(answer, truth))
    assert CROSS.cost(0.5, 1) == approx(math.log(2))
    assert CROSS.cost(0.99, 1) == approx(0.01, abs=0.001)
    assert CROSS.cost(0.01, 1) == approx(4.6, abs=0.01)
    assert CROSS.cost(0.01, 0) == CROSS.cost(0.99, 1)


def test_the_total_adds_the_cost_of_every_answer(show):
    answers, truths = [0, 3], [1, 1]
    show("costs", [SQUARED.cost(a, t) for a, t in zip(answers, truths)], "-> total", SQUARED.total(answers, truths))
    assert SQUARED.total(answers, truths) == 5


def test_every_slope_matches_a_tiny_nudge(show):
    nudge = 1e-6
    for loss in (SQUARED, CROSS):
        for answer, truth in ((0.3, 1), (0.8, 0), (0.59, 1)):
            felt = (loss.cost(answer + nudge, truth) - loss.cost(answer - nudge, truth)) / (2 * nudge)
            assert loss.slope(answer, truth) == approx(felt, abs=1e-5)
            show(type(loss).__name__, "answer", answer, "truth", truth,
                 "-> slope", loss.slope(answer, truth), "  nudge test", felt)
