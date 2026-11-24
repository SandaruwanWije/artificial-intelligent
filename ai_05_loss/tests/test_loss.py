"""The numbers the article quotes, checked against the code."""

from pytest import approx

from ai_05_loss.article import (
    ANSWER, BEST_LINE, CROSS, FAR_UP_THE_SIDE, LINE_MAP, POOR_LINE, RULE_MAPS, SQUARED, THREE_SPOTS, TRUTH,
)
from ai_05_loss.examples import TEN_POINTS
from ai_05_loss.models import Rule


def test_the_answer_is_0_59_and_misses_by_0_41():
    assert round(ANSWER, 2) == 0.59
    assert round(SQUARED.miss(ANSWER, TRUTH), 2) == 0.41


def test_one_answer_costs_0_17_in_squared_error():
    assert round(SQUARED.cost(ANSWER, TRUTH), 2) == 0.17


def test_twice_the_miss_costs_four_times_as_much():
    small, big = SQUARED.cost(answer=0, truth=0.3), SQUARED.cost(answer=0, truth=0.6)
    assert (round(small, 2), round(big, 2)) == (0.09, 0.36)
    assert big / small == approx(4)


def test_the_poor_line_has_a_mean_squared_error_of_0_47():
    assert round(SQUARED.average(POOR_LINE, TEN_POINTS), 2) == 0.47


def test_the_worst_misses_above_and_below_the_poor_line():
    misses = [POOR_LINE.miss(p) for p in TEN_POINTS]
    assert (round(max(misses), 2), round(min(misses), 2)) == (0.78, -0.96)


def test_the_best_line_has_a_mean_squared_error_of_0_11():
    assert (round(BEST_LINE.slope, 2), round(BEST_LINE.crossing, 2)) == (0.86, 0.32)
    assert round(SQUARED.average(BEST_LINE, TEN_POINTS), 2) == 0.11


def test_sure_and_wrong_costs_0_98_against_4_6():
    assert round(SQUARED.cost(0.01, 1), 2) == 0.98
    assert round(CROSS.cost(0.01, 1), 1) == 4.6


def test_surprise_for_each_bet():
    surprises = [CROSS.cost(bet, 1) for bet in (0.99, 0.59, 0.1, 0.01)]
    assert [round(s, 2) for s in surprises[:2]] == [0.01, 0.53]
    assert [round(s, 1) for s in surprises[2:]] == [2.3, 4.6]


def test_each_halving_of_the_share_adds_0_69():
    costs = [round(CROSS.cost(share, 1), 2) for share in (1, 0.5, 0.25, 0.125)]
    assert costs == [0, 0.69, 1.39, 2.08]


def test_the_same_answer_costs_0_17_and_0_53():
    assert (round(SQUARED.cost(ANSWER, TRUTH), 2), round(CROSS.cost(ANSWER, TRUTH), 2)) == (0.17, 0.53)


def test_the_chart_starts_at_0_96_and_3_91():
    assert (round(SQUARED.cost(0.02, 1), 2), round(CROSS.cost(0.02, 1), 2)) == (0.96, 3.91)


def test_squared_error_never_goes_above_1():
    assert all(SQUARED.cost(answer / 100, truth) <= 1 for answer in range(101) for truth in (0, 1))


def test_the_valley_is_lowest_at_a_slope_of_0_86():
    assert round(LINE_MAP.row_at(BEST_LINE.crossing).lowest().model.slope, 2) == 0.86


def test_three_spots_on_the_map():
    assert [round(LINE_MAP.at(line).loss, 2) for line in THREE_SPOTS] == [0.73, 1.46, 1.12]


def test_far_up_the_side_the_loss_is_14_08():
    assert round(LINE_MAP.at(FAR_UP_THE_SIDE).loss, 2) == 14.08


def test_the_lowest_yes_or_no_rule():
    assert RULE_MAPS[CROSS].lowest().model == Rule(weight=2.33, bias=-0.15)
