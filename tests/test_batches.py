import random

from pytest import approx

from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.batches import Batches, BatchSlopes, Shuffle
from neural_network.loss import SquaredError

BATCH_SLOPES = BatchSlopes(Backpropagation(SquaredError()))


def test_shuffle_keeps_every_mango_and_leaves_the_crate_as_it_was(show):
    crate = list(range(24))
    mixed = Shuffle(random.Random(5)).shuffled(crate)
    show("the crate   ", crate)
    show("shuffled    ", mixed)
    assert mixed[:6] == [22, 13, 9, 21, 18, 10]
    assert sorted(mixed) == crate
    assert crate == list(range(24))


def test_the_same_seed_gives_the_same_order_and_each_shuffle_is_a_new_mix(show):
    shuffle = Shuffle(random.Random(5))
    first = shuffle.shuffled(list(range(24)))
    second = shuffle.shuffled(list(range(24)))
    again = Shuffle(random.Random(5)).shuffled(list(range(24)))
    show("first shuffle       ", first)
    show("second shuffle      ", second)
    show("seed 5 from the start", again)
    assert again == first
    assert second != first


def test_24_mangoes_in_batches_of_6_are_4_batches(show):
    batches = Batches(6).cut(list(range(24)))
    for b in range(len(batches)):
        show("batch", b, batches[b])
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2, 3, 4, 5]
    assert batches[3] == [18, 19, 20, 21, 22, 23]


def test_what_is_left_over_makes_a_smaller_last_batch(show):
    batches = Batches(4).cut(list(range(10)))
    show("10 examples in batches of 4:", batches)
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_a_batch_slope_is_the_average_of_its_mangoes_slopes(crate, crate_network, show):
    batch = crate[:6]
    number = crate_network.layers[0].neurons[0].bias
    total = 0.0
    for inputs, truths in batch:
        own_slope = 2 * (number - truths[0])
        total = total + own_slope
        show("the number", number, "  this mango wants", truths[0], "  its slope 2 x miss =", own_slope)
        assert BATCH_SLOPES.slopes(crate_network, [(inputs, truths)])[0][0].bias == approx(own_slope)
    slope = BATCH_SLOPES.slopes(crate_network, batch)[0][0].bias
    show("total", total, "  / 6 mangoes = batch slope", slope)
    assert slope == approx(total / 6)
    assert slope == approx(-6.8867, abs=0.0001)


def test_the_whole_crate_reports_a_slope_of_minus_6_4(crate, crate_network, show):
    slope = BATCH_SLOPES.slopes(crate_network, crate)[0][0].bias
    show("first 6 mangoes: slope", BATCH_SLOPES.slopes(crate_network, crate[:6])[0][0].bias)
    show("all 24 mangoes:  slope", slope)
    assert slope == approx(-6.4)


def test_batch_slopes_match_the_average_of_the_nudged_slopes(mango_network, show):
    batch = [([1.0, 0.5], [1.0]), ([0.2, 0.9], [0.0]), ([0.7, 0.3], [1.0])]
    nudged = [FiniteDifference(SquaredError()).slopes(mango_network, inputs, truths) for inputs, truths in batch]
    swept = BATCH_SLOPES.slopes(mango_network, batch)
    for layer_number in range(len(swept)):
        for n in range(len(swept[layer_number])):
            weights = swept[layer_number][n].weights
            for i in range(len(weights)):
                average = (nudged[0][layer_number][n].weights[i] + nudged[1][layer_number][n].weights[i]
                           + nudged[2][layer_number][n].weights[i]) / 3
                assert weights[i] == approx(average, abs=1e-5)
            average = (nudged[0][layer_number][n].bias + nudged[1][layer_number][n].bias
                       + nudged[2][layer_number][n].bias) / 3
            show("layer", layer_number, "neuron", n, "weights", weights, "bias", swept[layer_number][n].bias,
                 "  nudged bias", average)
            assert swept[layer_number][n].bias == approx(average, abs=1e-5)
