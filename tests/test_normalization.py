import math
import random

from pytest import approx

from neural_network.activations import NO_BEND, RELU, SIGMOID, TANH
from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.batches import Shuffle
from neural_network.forward_pass import ForwardPass
from neural_network.gradient_descent import GradientDescent
from neural_network.initialization import Initialization, Xavier
from neural_network.loss import SquaredError
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron
from neural_network.normalization import LayerNorm
from neural_network.training import Trainer

EIGHT = [60.2, 41.6, 24.7, 57.1, 30.4, 48.0, 54.6, 44.9]


def small_network() -> Network:
    """2 inputs, 3 sums, a layer norm that bends with tanh, 1 sigmoid answer."""
    sums = Layer([Neuron([0.8, -0.4], 0.1, NO_BEND), Neuron([0.3, 0.9], -0.2, NO_BEND), Neuron([-0.6, 0.5], 0.3, NO_BEND)])
    return Network([sums, LayerNorm(3, TANH), Layer([Neuron([1.2, -0.7, 0.5], -0.1, SIGMOID)])])


def ten_normed_layers(start: Initialization, numbers: random.Random) -> Network:
    """Ten times: 16 sums, then a layer norm that bends them with ReLU. Then 1 sigmoid answer."""
    layers, inputs = [], 2
    for _ in range(10):
        layers = layers + [start.layer(inputs, 16, NO_BEND), LayerNorm(16, RELU)]
        inputs = 16
    return Network(layers + [Xavier(numbers).layer(16, 1, SIGMOID)])


def usual_size(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def test_standardizing_subtracts_the_mean_and_divides_by_the_spread(show):
    norm = LayerNorm(8, NO_BEND)
    mean, spread, standardized = norm.mean(EIGHT), norm.spread(EIGHT), norm.standardized(EIGHT)
    show("the numbers   ", EIGHT, "  mean", mean, "  spread", spread)
    show("minus the mean", [number - mean for number in EIGHT])
    show("/ the spread  ", standardized)
    assert (mean, spread) == (approx(45.19, abs=0.005), approx(11.81, abs=0.005))
    assert [round(number, 2) for number in standardized] == [1.27, -0.3, -1.74, 1.01, -1.25, 0.24, 0.8, -0.02]
    assert norm.mean(standardized) == approx(0, abs=1e-9)
    assert norm.spread(standardized) == approx(1)


def test_numbers_1024_times_smaller_land_on_the_same_places(show):
    norm = LayerNorm(8, NO_BEND)
    small = [number / 1024 for number in EIGHT]
    show("big  ", EIGHT, "->", norm.standardized(EIGHT))
    show("small", small, "->", norm.standardized(small))
    assert norm.standardized(small) == approx(norm.standardized(EIGHT), abs=1e-4)


def test_layer_norm_standardizes_one_mango_over_the_neurons_of_the_layer(show):
    one_mango_eight_neurons = [0.85, 0.34, 0.03, 0.41, -0.88, 0.82, -0.27, -0.21]
    norm = LayerNorm(8, NO_BEND)
    outputs = norm.outputs(one_mango_eight_neurons)
    show("one mango's eight numbers", one_mango_eight_neurons)
    show("mean", norm.mean(one_mango_eight_neurons), "  spread", norm.spread(one_mango_eight_neurons))
    show("out of the layer", outputs, " scale 1 and shift 0 change nothing yet")
    assert norm.mean(one_mango_eight_neurons) == approx(0.14, abs=0.005)
    assert norm.spread(one_mango_eight_neurons) == approx(0.55, abs=0.005)
    assert outputs == approx(norm.standardized(one_mango_eight_neurons))


def test_every_neuron_has_a_scale_that_starts_at_1_and_a_shift_that_starts_at_0(show):
    norm = LayerNorm(3, NO_BEND)
    assert [(neuron.weights, neuron.bias) for neuron in norm.neurons] == [([1.0], 0.0), ([1.0], 0.0), ([1.0], 0.0)]
    standardized = norm.standardized([3.0, 5.0, 10.0])
    norm.neurons[0].weights[0] = 2.0  # the scale of the first neuron
    norm.neurons[0].bias = 0.5  # and its shift
    show("standardized", standardized, "  first neuron: x 2, + 0.5 ->", norm.outputs([3.0, 5.0, 10.0]))
    assert norm.outputs([3.0, 5.0, 10.0]) == approx([2.0 * standardized[0] + 0.5, standardized[1], standardized[2]])


def test_the_forward_pass_goes_through_the_norm_like_the_network_does(mango, show):
    network, (inputs, _) = small_network(), mango
    sums, normed, answer = ForwardPass(network, inputs).steps
    show("the sums", sums.outputs, "  standardized, then tanh:", normed.outputs, "  answer", answer.outputs)
    assert sums.outputs == approx([0.7, 0.55, -0.05])
    assert normed.totals == approx(network.layers[1].standardized([0.7, 0.55, -0.05]))
    assert answer.outputs == approx(network.answer(inputs))  # the same in training and in use, for one mango


def test_every_slope_matches_a_nudge_also_for_each_scale_and_shift(mango, show):
    network = small_network()
    network.layers[1].neurons[0].weights[0] = 1.3  # so that not every scale is 1
    network.layers[1].neurons[1].bias = 0.2
    swept = Backpropagation(SquaredError()).slopes(network, *mango)
    nudged = FiniteDifference(SquaredError()).slopes(network, *mango)
    for n in range(3):
        show("norm neuron", n, " scale slope", swept[1][n].weights[0], "nudged", nudged[1][n].weights[0],
             "  shift slope", swept[1][n].bias, "nudged", nudged[1][n].bias)
    for layer_number in range(3):
        for n in range(len(swept[layer_number])):
            assert swept[layer_number][n].weights == approx(nudged[layer_number][n].weights, abs=1e-5)
            assert swept[layer_number][n].bias == approx(nudged[layer_number][n].bias, abs=1e-5)
    show("the sums behind the norm: swept", swept[0][0].weights, "  nudged", nudged[0][0].weights)


def test_with_a_norm_in_every_layer_the_signal_is_the_same_for_every_start(labelled_crate, times_he, show):
    signals = []
    for name, times in (("half of He", 0.5), ("He", 1.0), ("twice He", 2.0)):
        numbers = random.Random(3)
        network = ten_normed_layers(times_he(numbers, times), numbers)
        passes = [ForwardPass(network, inputs) for inputs, _ in labelled_crate]
        signal = [usual_size([output for one in passes for output in one.steps[step].outputs]) for step in (1, 7, 13, 19)]
        show(name, " signal after the norm of layers 1, 4, 7, 10:", signal)
        signals.append(signal)
    assert signals[0] == approx([0.691, 0.713, 0.797, 0.774], abs=0.001)
    assert signals[1] == approx(signals[0], abs=1e-4)
    assert signals[2] == approx(signals[0], abs=1e-4)


def test_with_layer_norm_all_three_starts_learn_and_the_scales_and_shifts_move(labelled_crate, times_he, show):
    for name, times in (("half of He", 0.5), ("He", 1.0), ("twice He", 2.0)):
        numbers = random.Random(3)
        network = ten_normed_layers(times_he(numbers, times), numbers)
        trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.05))
        losses = trainer.train(labelled_crate, epochs=3, batch_size=6, shuffle=Shuffle(random.Random(5)))
        learned = network.layers[9].neurons[9]
        show(name, " loss after 0, 1, 2, 3 epochs:", losses,
             "  one neuron's scale", learned.weights[0], "and shift", learned.bias)
        assert losses[3] < losses[0] - 0.02  # with no norm, the first and the last start stay stuck
        assert learned.weights[0] != 1.0
        assert learned.bias != 0.0
