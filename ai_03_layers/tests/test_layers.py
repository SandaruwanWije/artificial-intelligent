"""The numbers the article quotes, checked against the code."""

from functools import cache

from ai_03_layers.__main__ import CLOUD, TRAINER, TRIED
from ai_03_layers.tents import DeepOrWide, deep_tents
from ai_03_layers.trainer import Trial
from ai_03_layers.xor import (CORNERS, LINES_ONE_NEURON_TRIES, OUTPUT_LINE, XOR_NETWORK,
                              band_answer, corners_right, every_line_on_a_grid)


@cache
def trial(i: int) -> Trial:
    hidden_neurons, seed = TRIED[i]
    return Trial(hidden_neurons, seed, TRAINER.train(CLOUD, hidden_neurons, seed))


def test_xor_wants_1_when_exactly_one_input_is_1():
    assert [corner.wants for corner in CORNERS] == [0, 1, 1, 0]


def test_no_single_line_gets_more_than_3_of_4():
    assert max(corners_right(neuron) for neuron in LINES_ONE_NEURON_TRIES) == 3
    assert max(corners_right(neuron) for neuron in every_line_on_a_grid()) == 3


def test_two_lines_at_half_and_one_and_a_half_do_xor():
    assert [corner.x1 + corner.x2 for corner in CORNERS] == [0, 1, 1, 2]
    assert [band_answer(corner) for corner in CORNERS] == [0, 1, 1, 0]


def test_h1_and_h2_at_every_corner():
    new_numbers = [XOR_NETWORK.new_numbers(corner.inputs) for corner in CORNERS]
    assert new_numbers == [(0, 0), (1, 0), (1, 0), (2, 1)]


def test_the_network_answers_exactly_xor():
    assert [XOR_NETWORK.answer(corner.inputs) for corner in CORNERS] == [0, 1, 1, 0]


def test_one_line_splits_the_new_numbers():
    output = XOR_NETWORK.output_neuron
    for corner in CORNERS:
        above = output.weighted_sum(XOR_NETWORK.new_numbers(corner.inputs)) > OUTPUT_LINE
        assert above == bool(corner.wants)


def test_the_cloud_is_160_points_in_four_clusters_of_40():
    assert len(CLOUD) == 160
    for want_x1, want_x2 in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cluster = CLOUD[(want_x1 > 0) + 2 * (want_x2 > 0)::4]
        assert len(cluster) == 40
        assert all(p.wants == (want_x1 != want_x2) for p in cluster)


def test_one_hidden_neuron_gets_74_percent():
    assert trial(0).percent_right(CLOUD) == 74


def test_two_hidden_neurons_also_get_74_percent_with_one_switched_off():
    two = trial(1)
    assert two.percent_right(CLOUD) == 74
    [off] = two.switched_off()
    assert all(abs(weight) < 0.25 for weight in off.weights) and off.bias < 0


def test_the_same_two_neurons_from_another_start_get_97_percent():
    again = trial(2)
    assert again.seed != trial(1).seed
    assert again.percent_right(CLOUD) == 97
    assert len(again.lines_switched_on()) == 2


def test_three_four_and_eight_hidden_neurons_get_every_point():
    assert [trial(i).percent_right(CLOUD) for i in (3, 4, 5)] == [100, 100, 100]


def test_the_three_neuron_output_bias_is_7_22():
    assert round(trial(3).network.output_neuron.bias, 2) == 7.22


def test_the_tent_peaks_at_half_and_comes_back_to_0():
    assert [deep_tents(1).answer((x,)) for x in (0, 0.5, 1)] == [0, 1, 0]


def test_two_layers_make_two_tents():
    assert [deep_tents(2).answer((x,)) for x in (0, 0.25, 0.5, 0.75, 1)] == [0, 1, 0, 1, 0]


def test_corners_double_layer_by_layer():
    rows = [DeepOrWide.for_layers(layers) for layers in (1, 2, 3, 4)]
    assert [row.corners for row in rows] == [1, 3, 7, 15]
    assert [row.deep_neurons for row in rows] == [2, 4, 6, 8]


def test_ten_layers_take_20_neurons_against_1023():
    ten = DeepOrWide.for_layers(10)
    assert (ten.deep_neurons, ten.wide_neurons) == (20, 1023)
