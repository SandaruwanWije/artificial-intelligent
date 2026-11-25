from pytest import approx

from neural_network.activations import RELU, SIGMOID
from neural_network.backpropagation import Backpropagation
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron
from neural_network.training import Trainer

TRAINER = Trainer(Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))


def test_one_step_moves_w1_to_0_92_and_the_loss_to_0_075(mango_network, mango, show):
    first = mango_network.layers[0].neurons[0]
    show("before: first weight", first.weights[0], "  loss", TRAINER.loss(mango_network, [mango]))
    TRAINER.step(mango_network, *mango)
    show("after:  first weight", first.weights[0], "  loss", TRAINER.loss(mango_network, [mango]))
    assert round(mango_network.layers[0].neurons[0].weights[0], 2) == 0.92
    assert TRAINER.loss(mango_network, [mango]) == approx(0.075, abs=0.0005)


def test_40_rounds_end_96_percent_sure(mango_network, mango, show):
    losses = TRAINER.train(mango_network, [mango], rounds=40)
    for r in (0, 1, 2, 10, 40):
        show("after round", r, "  loss", losses[r])
    inputs, _ = mango
    show("the answer is now", mango_network.answer(inputs))
    assert [round(losses[r], 4) for r in (0, 1, 2, 10, 40)] == [0.1699, 0.0747, 0.0427, 0.0074, 0.0014]
    assert round(mango_network.answer(inputs)[0], 2) == 0.96


def test_training_learns_xor_from_a_start_that_cannot(show):
    network = Network([
        Layer([Neuron([0.5, 0.4], 0.0, RELU), Neuron([0.6, 0.5], -0.6, RELU)]),
        Layer([Neuron([0.3, -0.2], 0.1, SIGMOID)]),
    ])
    corners = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    losses = TRAINER.train(network, corners, rounds=2000)
    for r in (0, 10, 50, 100, 2000):
        show("after round", r, "  loss", losses[r])
    for inputs, truths in corners:
        show(inputs, "-> answer", network.answer(inputs), "  want", truths)
    assert losses[-1] < losses[0] / 10
    assert [round(network.answer(inputs)[0]) for inputs, _ in corners] == [0, 1, 1, 0]
