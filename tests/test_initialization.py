import math
import random

from pytest import approx

from neural_network.activations import NO_BEND, RELU, SIGMOID
from neural_network.backpropagation import Backpropagation
from neural_network.batches import Shuffle
from neural_network.forward_pass import ForwardPass
from neural_network.gradient_descent import GradientDescent
from neural_network.initialization import He, Initialization, Xavier
from neural_network.loss import SquaredError
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron
from neural_network.training import Trainer


def usual_size(values: list[float]) -> float:
    """The root of the mean of the squares."""
    return math.sqrt(sum(value * value for value in values) / len(values))


def ten_layers(start: Initialization) -> Network:
    """2 inputs, then ten layers of 16 ReLU neurons."""
    return Network([start.layer(2, 16, RELU)] + [start.layer(16, 16, RELU) for _ in range(9)])


def train(network: Network, crate: list, epochs: int, learning_rate: float) -> list[float]:
    trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate))
    return trainer.train(crate, epochs=epochs, batch_size=6, shuffle=Shuffle(random.Random(5)))


def test_the_spread_is_set_by_the_number_of_inputs(show):
    xavier, he = Xavier(random.Random(3)), He(random.Random(3))
    for inputs in (4, 16, 100):
        show(inputs, "inputs:  Xavier 1 / root n =", xavier.spread(inputs), "   He root 2 / root n =", he.spread(inputs))
    assert (xavier.spread(4), xavier.spread(16), xavier.spread(100)) == approx((0.5, 0.25, 0.1))
    assert he.spread(16) == approx(0.3536, abs=0.0001)
    assert he.spread(16) == approx(math.sqrt(2) * xavier.spread(16))


def test_a_layer_starts_with_random_weights_and_biases_of_0(show):
    layer = He(random.Random(3)).layer(inputs=2, neurons=3, activation=RELU)
    for neuron in layer.neurons:
        show("weights", neuron.weights, "  bias", neuron.bias)
    weights = [weight for neuron in layer.neurons for weight in neuron.weights]
    again = He(random.Random(3)).layer(inputs=2, neurons=3, activation=RELU)
    assert [len(neuron.weights) for neuron in layer.neurons] == [2, 2, 2]
    assert [neuron.bias for neuron in layer.neurons] == [0.0, 0.0, 0.0]
    assert len(set(weights)) == 6  # no two weights start the same
    assert weights == [weight for neuron in again.neurons for weight in neuron.weights]  # the same seed, the same start


def test_4000_weights_have_the_spread_the_rule_asks_for(show):
    measured = []
    for rule in (Xavier(random.Random(5)), He(random.Random(5))):
        layer = rule.layer(inputs=16, neurons=250, activation=RELU)
        size = usual_size([weight for neuron in layer.neurons for weight in neuron.weights])
        show(type(rule).__name__, " asks for", rule.spread(16), "  the 4000 weights have", size)
        measured.append(size)
    assert measured == approx([0.25, 0.3536], abs=0.01)


def test_a_sum_of_n_inputs_is_root_n_big_and_xaviers_spread_brings_it_back_to_1(times_he, show):
    for inputs, root in ((4, 2), (16, 4), (100, 10)):
        numbers = random.Random(5)
        spread_of_1 = times_he(numbers, math.sqrt(inputs) / math.sqrt(2)).layer(inputs, 4000, NO_BEND)
        sized = Xavier(numbers).layer(inputs, 4000, NO_BEND)
        plain = usual_size([neuron.total([numbers.gauss(0, 1) for _ in range(inputs)]) for neuron in spread_of_1.neurons])
        fixed = usual_size([neuron.total([numbers.gauss(0, 1) for _ in range(inputs)]) for neuron in sized.neurons])
        show(inputs, "inputs: with a spread of 1 the sum is", plain, "  with Xavier's spread it is", fixed)
        assert plain == approx(root, rel=0.05)
        assert fixed == approx(1, rel=0.05)


def test_only_the_right_spread_keeps_the_signal_steady_through_ten_layers(labelled_crate, times_he, show):
    last_over_first = []
    starts = (times_he(random.Random(3), 0.5), He(random.Random(3)), times_he(random.Random(3), 2), Xavier(random.Random(3)))
    for name, start in zip(("half of He", "He", "twice He", "Xavier"), starts):
        network = ten_layers(start)
        passes = [ForwardPass(network, inputs) for inputs, _ in labelled_crate]
        signal = [usual_size([output for one in passes for output in one.steps[layer].outputs]) for layer in range(10)]
        show(name, " spread", start.spread(16), "  signal at layers 1, 4, 7, 10:", [signal[layer] for layer in (0, 3, 6, 9)])
        last_over_first.append(signal[9] / signal[0])
    assert last_over_first[0] < 1 / 300  # too small: almost nothing reaches the end
    assert 0.5 < last_over_first[1] < 2  # He: about as strong as it started
    assert last_over_first[2] > 300  # too big: it grows at every layer
    assert last_over_first[3] < 1 / 15  # Xavier's spread is too small for ReLU


def test_neurons_that_start_the_same_stay_the_same(labelled_crate, show):
    network = Network([Layer([Neuron([0.5, 0.5], 0.5, RELU) for _ in range(3)]), Layer([Neuron([0.5] * 3, 0.5, SIGMOID)])])
    losses = train(network, labelled_crate, epochs=300, learning_rate=0.5)
    for neuron in network.layers[0].neurons:
        show("after 300 epochs: weights", neuron.weights, "  bias", neuron.bias)
    show("loss", losses[0], "->", losses[300])
    first, second, third = network.layers[0].neurons
    assert first.weights == second.weights == third.weights
    assert first.bias == second.bias == third.bias


def test_weights_that_start_at_0_never_move(labelled_crate, show):
    network = Network([Layer([Neuron([0.0, 0.0], 0.0, RELU) for _ in range(3)]), Layer([Neuron([0.0] * 3, 0.0, SIGMOID)])])
    losses = train(network, labelled_crate, epochs=300, learning_rate=0.5)
    weights = [neuron.weights for layer in network.layers for neuron in layer.neurons]
    show("after 300 epochs every weight is still", weights, "  loss", losses[0], "->", losses[300])
    assert weights == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0]]
    assert losses[300] == approx(0.25, abs=0.001)  # it answers about 0.5 for every mango


def test_a_random_start_gives_every_neuron_its_own_weights(labelled_crate, show):
    numbers = random.Random(3)
    network = Network([He(numbers).layer(2, 3, RELU), Xavier(numbers).layer(3, 1, SIGMOID)])
    losses = train(network, labelled_crate, epochs=300, learning_rate=0.5)
    for neuron in network.layers[0].neurons:
        show("after 300 epochs: weights", neuron.weights, "  bias", neuron.bias)
    show("loss", losses[0], "->", losses[300])
    first, second, third = network.layers[0].neurons
    assert first.weights != second.weights != third.weights != first.weights
    assert losses[300] == approx(0.1056, abs=0.0001)


def test_a_deep_network_only_learns_from_the_right_size_of_start(labelled_crate, times_he, show):
    ends = []
    for name, times in (("half of He", 0.5), ("He", 1.0), ("twice He", 2.0)):
        numbers = random.Random(3)
        network = Network(ten_layers(times_he(numbers, times)).layers + [Xavier(numbers).layer(16, 1, SIGMOID)])
        losses = train(network, labelled_crate, epochs=10, learning_rate=0.05)
        show(name, " loss after 0, 2, 5, 10 epochs:", [losses[epoch] for epoch in (0, 2, 5, 10)])
        ends.append((round(losses[0], 4), round(losses[10], 4)))
    assert ends == [(0.25, 0.25), (0.2696, 0.2261), (0.5, 0.5)]  # stuck, learning, stuck
