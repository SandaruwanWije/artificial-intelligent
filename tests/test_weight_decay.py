import random

from pytest import approx

from neural_network.activations import RELU, SIGMOID
from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.batches import BatchSlopes, Shuffle
from neural_network.gradient_clipping import NO_LIMIT
from neural_network.gradient_descent import GradientDescent
from neural_network.initialization import He, Xavier
from neural_network.loss import SquaredError
from neural_network.network import Network
from neural_network.splits import Split
from neural_network.training import Trainer
from neural_network.weight_decay import NO_DECAY, WeightDecay

BATCH_SLOPES = BatchSlopes(Backpropagation(SquaredError()))


def squares_of(network: Network) -> float:
    return sum(weight * weight for layer in network.layers for neuron in layer.neurons for weight in neuron.weights)


def test_the_penalty_is_half_the_strength_times_every_weight_squared(mango_network, show):
    decay = WeightDecay(strength=0.3)
    show("every weight squared, added up:", squares_of(mango_network), "  x 0.5 x 0.3 =", decay.penalty(mango_network))
    assert squares_of(mango_network) == approx(4.49)
    assert decay.penalty(mango_network) == approx(0.6735)
    mango_network.layers[0].neurons[0].bias = 100.0  # biases are free
    assert decay.penalty(mango_network) == approx(0.6735)


def test_every_weight_slope_gets_the_strength_times_the_weight_added(mango_network, mango, show):
    slopes = BATCH_SLOPES.slopes(mango_network, [mango])
    added = WeightDecay(strength=0.3).added(mango_network, slopes)
    first = mango_network.layers[0].neurons[0]
    show("weights", first.weights, "  slopes from the data", slopes[0][0].weights)
    show("pull = 0.3 x weight =", [0.3 * weight for weight in first.weights], "  slopes with the pull", added[0][0].weights)
    assert added[0][0].weights == approx([-0.2397 + 0.24, -0.1198 - 0.12], abs=0.0001)
    assert added[0][0].bias == slopes[0][0].bias  # the bias slope is left as it was
    assert NO_DECAY.added(mango_network, slopes)[0][0].weights == slopes[0][0].weights


def test_the_added_slope_is_the_slope_of_the_penalty_checked_by_a_nudge(mango_network, mango, show):
    decay, nudge = WeightDecay(strength=0.3), FiniteDifference(SquaredError())
    slopes = BATCH_SLOPES.slopes(mango_network, [mango])
    added = decay.added(mango_network, slopes)
    for layer_number in range(len(mango_network.layers)):
        for n in range(len(mango_network.layers[layer_number].neurons)):
            neuron = mango_network.layers[layer_number].neurons[n]
            for i in range(len(neuron.weights)):
                before = decay.penalty(mango_network)
                neuron.weights[i] = neuron.weights[i] + nudge.nudge
                nudged = nudge.slope(before, decay.penalty(mango_network))
                neuron.weights[i] = neuron.weights[i] - nudge.nudge
                show("weight", neuron.weights[i], "  the penalty's slope by a nudge", nudged, "  added", 0.3 * neuron.weights[i])
                assert added[layer_number][n].weights[i] - slopes[layer_number][n].weights[i] == approx(nudged, abs=1e-5)


def test_a_weight_first_shrinks_then_takes_its_usual_step(mango_network, mango, show):
    trainer = Trainer(mango_network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5), NO_LIMIT,
                      WeightDecay(strength=0.3))
    slope = BATCH_SLOPES.slopes(mango_network, [mango])[0][0].weights[0]
    trainer.step([mango])
    show("weight 0.8   shrink 0.5 x 0.3 x 0.8 =", 0.5 * 0.3 * 0.8, "  usual step 0.5 x slope =", 0.5 * slope)
    show("new weight = 0.8 - 0.12 -", 0.5 * slope, "=", mango_network.layers[0].neurons[0].weights[0])
    assert mango_network.layers[0].neurons[0].weights[0] == approx(0.8 - 0.12 - 0.5 * slope)


def test_a_weight_the_data_does_not_push_decays_toward_0(mango_network, show):
    trainer = Trainer(mango_network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5), NO_LIMIT,
                      WeightDecay(strength=0.3))
    no_push = ([0.0, 0.0], [1.0])  # both inputs are 0, so every hidden weight's slope from the data is 0
    first = mango_network.layers[0].neurons[0]
    show("before:", first.weights)
    for step in range(10):
        trainer.step([no_push])
    show("after 10 steps, each one x (1 - 0.5 x 0.3) = 0.85:", first.weights, "  0.85 ten times over is", 0.85 ** 10)
    assert first.weights == approx([0.8 * 0.85 ** 10, -0.4 * 0.85 ** 10])


def test_more_decay_means_smaller_weights_and_too_much_crushes_them(labelled_crate, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    held, sizes = split.validation + split.test, []
    for strength in (0.0, 0.003, 0.3):
        numbers = random.Random(3)
        network = Network([He(numbers).layer(2, 20, RELU), Xavier(numbers).layer(20, 1, SIGMOID)])
        trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5), NO_LIMIT,
                          WeightDecay(strength))
        losses = trainer.train_and_validate(split.training, held, epochs=100, batch_size=6, shuffle=Shuffle(random.Random(5)))
        show("strength", strength, "  squared weights", squares_of(network), "  training loss", losses.training[100],
             "  validation loss", losses.validation[100])
        sizes.append((round(squares_of(network), 1), round(losses.training[100], 3), round(losses.validation[100], 3)))
    assert sizes == [(65.2, 0.083, 0.197), (29.8, 0.096, 0.179), (0.0, 0.248, 0.259)]
