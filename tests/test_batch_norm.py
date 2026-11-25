import math
import random

from pytest import approx

from neural_network.activations import NO_BEND, RELU, SIGMOID, TANH
from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.batches import BatchSlopes, Shuffle
from neural_network.forward_pass import BatchForwardPass
from neural_network.gradient_descent import GradientDescent
from neural_network.initialization import Xavier
from neural_network.loss import SquaredError
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron
from neural_network.normalization import BatchNorm, LayerNorm
from neural_network.training import Trainer

GRID = [  # six mangoes down, the sums of eight neurons across
    [0.85, 0.34, 0.03, 0.41, -0.88, 0.82, -0.27, -0.21],
    [-0.52, 0.49, 1.33, -0.16, -0.78, 0.53, -0.48, 0.36],
    [0.39, -1.24, -0.28, 0.01, 0.04, 0.48, -0.10, -0.18],
    [-3.25, 1.73, 2.94, -0.04, -1.87, -0.19, -0.62, 0.39],
    [-3.36, 1.92, 2.77, -0.24, -1.94, -0.41, -0.16, 0.03],
    [0.83, -0.64, -0.24, 0.83, -0.62, 0.76, -0.18, -0.53],
]
LOSS = SquaredError()


def small_network() -> Network:
    """2 inputs, 3 sums, a batch norm that bends with tanh, 1 sigmoid answer."""
    sums = Layer([Neuron([0.8, -0.4], 0.1, NO_BEND), Neuron([0.3, 0.9], -0.2, NO_BEND), Neuron([-0.6, 0.5], 0.3, NO_BEND)])
    return Network([sums, BatchNorm(3, TANH), Layer([Neuron([1.2, -0.7, 0.5], -0.1, SIGMOID)])])


def training_loss(network: Network, batch: list) -> float:
    """The batch's average loss as it is in training, with every norm measuring this batch."""
    forward = BatchForwardPass(network, [inputs for inputs, _ in batch])
    return sum(LOSS.total(forward.passes[e].answer, batch[e][1]) for e in range(len(batch))) / len(batch)


def test_batch_norm_goes_down_a_column_and_layer_norm_goes_along_a_row(show):
    batch_norm, layer_norm = BatchNorm(8, NO_BEND), LayerNorm(8, NO_BEND)
    batch_norm.see_inputs(GRID)
    show("neuron 1 over the six mangoes:", [row[0] for row in GRID], "  mean", batch_norm.means[0], " spread", batch_norm.spreads[0])
    show("mango 1 over the eight neurons:", GRID[0], "  mean", layer_norm.mean(GRID[0]), " spread", layer_norm.spread(GRID[0]))
    assert (batch_norm.means[0], batch_norm.spreads[0]) == (approx(-0.84, abs=0.005), approx(1.80, abs=0.005))
    assert (layer_norm.mean(GRID[0]), layer_norm.spread(GRID[0])) == (approx(0.14, abs=0.005), approx(0.55, abs=0.005))


def test_in_training_each_neuron_reads_its_input_standardized_over_the_batch(show):
    batch_norm = BatchNorm(8, NO_BEND)
    batch_norm.see_inputs(GRID)
    column = [batch_norm.inputs_of(0, row)[0] for row in GRID]
    show("neuron 1, standardized over the batch:", column)
    assert column == approx(batch_norm.standardized([row[0] for row in GRID]))
    assert batch_norm.mean(column) == approx(0, abs=1e-9)
    assert batch_norm.spread(column) == approx(1)


def test_a_batch_of_one_mango_leaves_nothing_so_use_needs_what_training_kept(show):
    batch_norm = BatchNorm(8, NO_BEND)
    batch_norm.see_inputs([GRID[0]])
    alone = [batch_norm.inputs_of(n, GRID[0])[0] for n in range(8)]
    show("one mango alone: every number minus its own mean is", alone)
    assert alone == approx([0.0] * 8)


def test_it_keeps_a_running_mean_and_spread_and_uses_them_when_the_network_is_used(show):
    batch_norm = BatchNorm(8, NO_BEND)
    batch_norm.see_inputs(GRID)
    show("after one batch: kept mean", batch_norm.kept_means[0], "= 0.9 x 0 + 0.1 x", batch_norm.means[0],
         "  kept spread", batch_norm.kept_spreads[0], "= 0.9 x 1 + 0.1 x", batch_norm.spreads[0])
    assert batch_norm.kept_means[0] == approx(0.1 * batch_norm.means[0])
    assert batch_norm.kept_spreads[0] == approx(0.9 + 0.1 * batch_norm.spreads[0])
    for _ in range(100):
        batch_norm.see_inputs(GRID)
    in_use = batch_norm.outputs(GRID[0])
    show("after 100 more: kept mean", batch_norm.kept_means[0], " kept spread", batch_norm.kept_spreads[0])
    show("one mango in use:", in_use)
    assert (batch_norm.kept_means[0], batch_norm.kept_spreads[0]) == (approx(-0.84, abs=0.005), approx(1.80, abs=0.005))
    assert in_use == approx([batch_norm.inputs_of(n, GRID[0])[0] for n in range(8)], abs=1e-4)


def test_every_averaged_slope_matches_a_nudge_of_the_whole_batchs_loss(labelled_crate, show):
    network, batch, nudge = small_network(), labelled_crate[:4], FiniteDifference(LOSS)
    network.layers[1].neurons[0].weights[0] = 1.3  # so that not every scale is 1
    network.layers[1].neurons[1].bias = 0.2
    slopes = BatchSlopes(Backpropagation(LOSS)).slopes(network, batch)
    for layer_number in range(3):
        for n in range(len(network.layers[layer_number].neurons)):
            neuron, nudged = network.layers[layer_number].neurons[n], []
            for i in range(len(neuron.weights)):
                before = training_loss(network, batch)
                neuron.weights[i] = neuron.weights[i] + nudge.nudge
                nudged.append(nudge.slope(before, training_loss(network, batch)))
                neuron.weights[i] = neuron.weights[i] - nudge.nudge
            show("layer", layer_number, "neuron", n, " swept", slopes[layer_number][n].weights, "  nudged", nudged)
            assert slopes[layer_number][n].weights == approx(nudged, abs=1e-5)
    sums_bias_slopes = [neuron_slopes.bias for neuron_slopes in slopes[0]]
    show("bias slopes of the sums before the norm:", sums_bias_slopes, " subtracting the mean removes a bias")
    assert sums_bias_slopes == approx([0.0, 0.0, 0.0], abs=1e-12)


def test_with_batch_norm_the_signal_is_the_same_and_all_three_starts_learn(labelled_crate, times_he, show):
    signals, ends = [], []
    for name, times in (("half of He", 0.5), ("He", 1.0), ("twice He", 2.0)):
        numbers, layers, inputs = random.Random(3), [], 2
        for _ in range(10):
            layers = layers + [times_he(numbers, times).layer(inputs, 16, NO_BEND), BatchNorm(16, RELU)]
            inputs = 16
        network = Network(layers + [Xavier(numbers).layer(16, 1, SIGMOID)])
        forward = BatchForwardPass(network, [mango_inputs for mango_inputs, _ in labelled_crate])
        outputs = [output for one in forward.passes for output in one.steps[19].outputs]
        signals.append(math.sqrt(sum(output * output for output in outputs) / len(outputs)))
        trainer = Trainer(network, Backpropagation(LOSS), GradientDescent(learning_rate=0.05))
        losses = trainer.train(labelled_crate, epochs=10, batch_size=6, shuffle=Shuffle(random.Random(5)))
        show(name, " signal after the tenth norm", signals[-1], "  loss in use after 0, 5, 10 epochs:",
             [losses[epoch] for epoch in (0, 5, 10)], "  loss in training", training_loss(network, labelled_crate))
        ends.append((round(losses[0], 2), round(losses[10], 2)))
    assert signals == approx([0.753, 0.753, 0.753], abs=0.001)
    assert ends == [(0.25, 0.2), (0.26, 0.23), (0.5, 0.16)]  # with no norm, the first and the last stay stuck
