import math
import random

from pytest import approx

from neural_network.activations import NO_BEND, RELU, SIGMOID, Activation
from neural_network.backpropagation import Backpropagation, FiniteDifference, NeuronSlopes
from neural_network.batches import BatchSlopes
from neural_network.gradient_clipping import NO_LIMIT, GradientClipping
from neural_network.gradient_descent import GradientDescent
from neural_network.initialization import He, Initialization, Xavier
from neural_network.loss import SquaredError
from neural_network.network import Network
from neural_network.training import Trainer

BATCH_SLOPES = BatchSlopes(Backpropagation(SquaredError()))
MEASURE = GradientClipping(limit=1.0)


def ten_layers(start: Initialization, activation: Activation, last: Initialization, answer: Activation) -> Network:
    """2 inputs, ten layers of 16 neurons, then 1 answer."""
    hidden = [start.layer(2, 16, activation)] + [start.layer(16, 16, activation) for _ in range(9)]
    return Network(hidden + [last.layer(16, 1, answer)])


def test_slopes_of_3600_and_4800_are_6000_long_and_clip_to_0_6_and_0_8(show):
    slopes = [[NeuronSlopes(weights=[3600.0, 4800.0], bias=0.0)]]
    clipped = MEASURE.clipped(slopes)
    show("slopes", slopes[0][0].weights, "  length", MEASURE.length(slopes))
    show("shrink by limit / length = 1 / 6000  ->", clipped[0][0].weights, "  length", MEASURE.length(clipped))
    assert MEASURE.length(slopes) == approx(6000)
    assert clipped[0][0].weights == approx([0.6, 0.8])  # still 3 to 4: the same direction
    assert MEASURE.length(clipped) == approx(1.0)


def test_slopes_shorter_than_the_limit_are_left_alone(show):
    slopes = [[NeuronSlopes(weights=[0.3, -0.4], bias=0.0)]]
    show("length", MEASURE.length(slopes), " is under the limit 1.0, so nothing changes")
    assert MEASURE.clipped(slopes) == slopes
    assert NO_LIMIT.clipped([[NeuronSlopes([3600.0, 4800.0], 0.0)]])[0][0].weights == [3600.0, 4800.0]


def test_the_length_counts_every_weight_and_bias_in_every_layer(mango_network, mango, show):
    slopes = BATCH_SLOPES.slopes(mango_network, [mango])
    squares = 0.0
    for layer in slopes:
        for neuron_slopes in layer:
            show("slopes", neuron_slopes.weights, "  bias slope", neuron_slopes.bias)
            squares = squares + sum(slope * slope for slope in neuron_slopes.weights) + neuron_slopes.bias * neuron_slopes.bias
    show("every slope squared and added up:", squares, "  its root, the length:", math.sqrt(squares))
    assert MEASURE.length(slopes) == approx(math.sqrt(squares))
    assert MEASURE.length(slopes) == approx(0.4947, abs=0.0001)


def test_clipped_slopes_are_the_nudged_slopes_all_shrunk_by_the_same_number(mango_network, mango, show):
    clipping = GradientClipping(limit=0.1)
    nudged = FiniteDifference(SquaredError()).slopes(mango_network, *mango)
    clipped = clipping.clipped(BATCH_SLOPES.slopes(mango_network, [mango]))
    shrink = 0.1 / clipping.length(nudged)
    show("nudged length", clipping.length(nudged), "  shrink", shrink, "  clipped length", clipping.length(clipped))
    for layer_number in range(len(nudged)):
        for n in range(len(nudged[layer_number])):
            wanted = [slope * shrink for slope in nudged[layer_number][n].weights]
            show("nudged x shrink", wanted, "  clipped", clipped[layer_number][n].weights)
            assert clipped[layer_number][n].weights == approx(wanted, abs=1e-6)
            assert clipped[layer_number][n].bias == approx(nudged[layer_number][n].bias * shrink, abs=1e-6)
    assert clipping.length(clipped) == approx(0.1)


def test_through_ten_sigmoid_layers_the_slopes_vanish(labelled_crate, show):
    numbers = random.Random(3)
    network = ten_layers(Xavier(numbers), SIGMOID, Xavier(numbers), SIGMOID)
    slopes = BATCH_SLOPES.slopes(network, labelled_crate)
    lengths = [MEASURE.length([slopes[layer]]) for layer in range(10)]
    for layer in (9, 7, 5, 3, 1, 0):
        show("slopes of layer", layer + 1, "are", f"{lengths[layer]:.2g}", "long")
    assert lengths[9] == approx(0.031, abs=0.001)
    assert lengths[0] < lengths[9] / 500_000  # almost nothing is left when it reaches the first layer


def test_weights_twice_too_big_make_the_gradient_a_million_long(labelled_crate, times_he, show):
    lengths = []
    for name, times in (("He", 1.0), ("twice He", 2.0)):
        numbers = random.Random(3)
        network = ten_layers(times_he(numbers, times), RELU, Xavier(numbers), NO_BEND)
        lengths.append(MEASURE.length(BATCH_SLOPES.slopes(network, labelled_crate)))
        show(name, " the whole gradient is", lengths[-1], "long")
    assert lengths[0] == approx(5.97, abs=0.01)  # healthy
    assert lengths[1] == approx(1_124_161, rel=0.001)  # exploding


def test_clipping_keeps_an_exploding_network_alive(labelled_crate, times_he, show):
    ends = []
    for clipping, steps in ((NO_LIMIT, 3), (GradientClipping(limit=1.0), 30)):
        numbers = random.Random(3)
        network = ten_layers(times_he(numbers, 2.0), RELU, Xavier(numbers), NO_BEND)
        trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.05), clipping)
        losses = [trainer.loss(labelled_crate)]
        for _ in range(steps):
            trainer.step(labelled_crate)
            losses.append(trainer.loss(labelled_crate))
        show("limit", clipping.limit, "  loss after 0, 1, 2, 3 steps:", [f"{loss:.4g}" for loss in losses[:4]],
             "  after", steps, "steps:", f"{losses[steps]:.4g}")
        ends.append(losses)
    assert ends[0][0] == approx(73132, rel=0.001)
    assert ends[0][1] > 1e70  # one step with no limit throws every weight far away
    assert max(ends[1][1:]) < ends[1][0] / 2  # with the limit, the loss never goes back up to the start
    assert ends[1][30] < ends[1][0] / 20


def test_a_clipped_step_moves_the_weights_by_the_learning_rate_times_the_limit(mango_network, mango, show):
    trainer = Trainer(mango_network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5),
                      GradientClipping(limit=0.1))
    before = [list(neuron.weights) + [neuron.bias] for layer in mango_network.layers for neuron in layer.neurons]
    trainer.step([mango])
    after = [list(neuron.weights) + [neuron.bias] for layer in mango_network.layers for neuron in layer.neurons]
    moved = math.sqrt(sum((after[n][i] - before[n][i]) * (after[n][i] - before[n][i])
                          for n in range(len(before)) for i in range(len(before[n]))))
    show("the gradient was 0.4947 long, the limit is 0.1, the learning rate is 0.5")
    show("all the weights and biases together moved", moved, "= 0.5 x 0.1")
    assert moved == approx(0.05)
