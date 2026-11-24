"""The numbers both parts of the article quote, checked against the code."""

from pytest import approx

from ai_07_backpropagation.__main__ import ARTICLE_7_NETWORK, MANGO, NUDGE_TEST, TRAINER
from ai_07_backpropagation.backprop import Backprop

PASSED = ARTICLE_7_NETWORK.forward(MANGO)
NUDGES = NUDGE_TEST.nudge_each(ARTICLE_7_NETWORK, MANGO)
BACKPROP = Backprop(ARTICLE_7_NETWORK, MANGO)
SWEEP = BACKPROP.sweep()
SLOPES = {"w1": -0.24, "w2": -0.12, "b1": -0.24, "w3": 0.14, "w4": 0.07, "b2": 0.14,
          "w5": 0, "w6": 0, "b3": 0, "w7": -0.14, "w8": -0.11, "w9": 0, "b4": -0.20}


def test_the_mango_gives_0_59_and_a_loss_of_0_17():
    assert (PASSED.h1, PASSED.h2, PASSED.h3) == approx((0.7, 0.55, 0))
    assert PASSED.y_sum == approx(0.355)
    assert round(PASSED.y, 2) == 0.59
    assert round(PASSED.loss, 2) == 0.17
    assert round(PASSED.loss, 4) == 0.1699


def test_the_network_can_change_13_numbers():
    names = ARTICLE_7_NETWORK.parameters.names()
    assert len(names) == 13
    assert sum(name.startswith("w") for name in names) == 9


def test_each_number_pushes_the_loss_the_way_the_article_says():
    loss_moves = {name: nudge.loss_moves for name, nudge in NUDGES.items()}
    assert {name for name, moves in loss_moves.items() if moves == "comes down"} == {
        "w1", "w2", "b1", "w7", "w8", "b4"}
    assert {name for name, moves in loss_moves.items() if moves == "goes up"} == {"w3", "w4", "b2"}
    assert NUDGES["w5"].road() == ["h3 sum"]
    assert NUDGES["w9"].road() == []


def test_nudging_w1_by_0_01_gives_a_slope_of_minus_0_24():
    nudge = NUDGES["w1"]
    assert nudge.after.h1 == approx(0.71)
    assert nudge.after.y_sum == approx(0.367)
    assert round(nudge.after.y, 3) == 0.591
    assert round(nudge.after.loss, 4) == 0.1675
    assert round(nudge.change_in_loss, 4) == -0.0024
    assert round(nudge.slope, 2) == -0.24


def test_hidden_numbers_have_five_links_and_the_four_into_y_have_three():
    roads = {name: BACKPROP.road(name) for name in SLOPES}
    assert [len(road) for road in roads.values()] == [5] * 9 + [3] * 4
    assert {(road[-2].top, road[-1].top) for road in roads.values()} == {("y", "loss")}


def test_the_five_links_of_w1():
    ratios = [link.ratio for link in BACKPROP.road("w1")]
    assert [round(ratio, 2) for ratio in ratios] == [1, 1, 1.2, 0.24, -0.82]


def test_every_slope_link_by_link_matches_the_table():
    for name, slope in SLOPES.items():
        assert round(BACKPROP.slope_link_by_link(name), 2) == approx(slope), name


def test_the_sweep_writes_the_articles_numbers():
    waiting_at = {part: round(number, 2) for part, number in SWEEP.waiting_at.items()}
    assert waiting_at == approx({"loss": 1, "y": -0.82, "y sum": -0.20, "h1": -0.24, "h2": 0.14,
                                 "h3": -0.10, "h1 sum": -0.24, "h2 sum": 0.14, "h3 sum": 0})


def test_nudge_links_and_sweep_agree_on_all_13_slopes():
    for name in SLOPES:
        assert SWEEP.slopes[name] == approx(BACKPROP.slope_link_by_link(name))
        assert NUDGES[name].slope == approx(SWEEP.slopes[name], abs=0.005), name


def test_one_step_moves_w1_to_0_92_and_the_loss_to_0_075():
    stepped = TRAINER.step(ARTICLE_7_NETWORK, MANGO)
    after = stepped.forward(MANGO)
    assert round(stepped.parameters.w1, 2) == 0.92
    assert (round(after.h1, 2), round(after.h2, 2), after.h3) == (0.97, 0.39, 0)
    assert round(after.y, 2) == 0.73
    assert round(after.loss, 3) == 0.075


def test_40_rounds_end_96_percent_sure():
    run = TRAINER.train(ARTICLE_7_NETWORK, MANGO)
    assert len(run.passes) == 41
    assert [round(run.losses[r], 4) for r in (0, 1, 2, 10, 40)] == [0.1699, 0.0747, 0.0427,
                                                                   0.0074, 0.0014]
    assert round(run.answers[-1], 2) == 0.96
