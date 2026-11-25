import random

from pytest import approx

from neural_network.activations import RELU, SIGMOID
from neural_network.backpropagation import Backpropagation
from neural_network.batches import Batches, Shuffle
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron
from neural_network.optimizers import Adam, Momentum, Optimizer, RMSProp
from neural_network.training import Trainer


def trainer_for(network: Network, optimizer: Optimizer) -> Trainer:
    return Trainer(network, Backpropagation(SquaredError()), optimizer)


def test_one_step_moves_w1_to_0_92_and_the_loss_to_0_075(mango_network, mango, show):
    trainer = trainer_for(mango_network, GradientDescent(learning_rate=0.5))
    first = mango_network.layers[0].neurons[0]
    show("before: first weight", first.weights[0], "  loss", trainer.loss([mango]))
    trainer.step([mango])
    show("after:  first weight", first.weights[0], "  loss", trainer.loss([mango]))
    assert round(first.weights[0], 2) == 0.92
    assert trainer.loss([mango]) == approx(0.075, abs=0.0005)


def test_40_epochs_on_one_mango_end_96_percent_sure(mango_network, mango, show):
    trainer = trainer_for(mango_network, GradientDescent(learning_rate=0.5))
    losses = trainer.train([mango], epochs=40, batch_size=1, shuffle=Shuffle(random.Random(5)))
    for e in (0, 1, 2, 10, 40):
        show("after epoch", e, "  loss", losses[e])
    inputs, _ = mango
    show("the answer is now", mango_network.answer(inputs))
    assert [round(losses[e], 4) for e in (0, 1, 2, 10, 40)] == [0.1699, 0.0747, 0.0427, 0.0074, 0.0014]
    assert round(mango_network.answer(inputs)[0], 2) == 0.96


def test_training_learns_xor_from_a_start_that_cannot(show):
    network = Network([
        Layer([Neuron([0.5, 0.4], 0.0, RELU), Neuron([0.6, 0.5], -0.6, RELU)]),
        Layer([Neuron([0.3, -0.2], 0.1, SIGMOID)]),
    ])
    corners = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    trainer = trainer_for(network, GradientDescent(learning_rate=0.5))
    losses = trainer.train(corners, epochs=2000, batch_size=1, shuffle=Shuffle(random.Random(5)))
    for e in (0, 10, 50, 100, 2000):
        show("after epoch", e, "  loss", losses[e])
    for inputs, truths in corners:
        show(inputs, "-> answer", network.answer(inputs), "  want", truths)
    assert losses[-1] < losses[0] / 10
    assert [round(network.answer(inputs)[0]) for inputs, _ in corners] == [0, 1, 1, 0]


def test_one_step_on_the_whole_crate_moves_the_number_0_64(crate, crate_network, show):
    trainer = trainer_for(crate_network, GradientDescent(learning_rate=0.1))
    neuron = crate_network.layers[0].neurons[0]
    slope = trainer.batch_slopes.slopes(crate_network, crate)[0][0].bias
    show("before: the number", neuron.bias, "  loss", trainer.loss(crate), "  slope", slope)
    show("step = learning rate x slope = 0.1 x", slope, "=", 0.1 * slope)
    assert trainer.loss(crate) == approx(11.2402, abs=0.0001)
    assert slope == approx(-6.4)
    trainer.step(crate)
    show("after:  the number", neuron.bias, "  loss", trainer.loss(crate))
    assert neuron.bias == approx(-1.56)
    assert trainer.loss(crate) == approx(7.5538, abs=0.0001)


def test_20_whole_crate_steps_at_three_learning_rates(crate, crate_network, show):
    after_20 = []
    for learning_rate in (0.01, 0.1, 1):
        crate_network.layers[0].neurons[0].bias = -2.2  # every run starts from the same place
        trainer = trainer_for(crate_network, GradientDescent(learning_rate))
        losses = trainer.train(crate, epochs=20, batch_size=24, shuffle=Shuffle(random.Random(5)))
        show("learning rate", learning_rate, "  loss after 0, 1, 2, 3 steps", losses[:4], "  after 20:", losses[20])
        after_20.append(losses[20])
    assert after_20[0] == approx(5.5642, abs=0.0001)  # too low: still far from the bottom
    assert after_20[1] == approx(1.0016, abs=0.0001)  # about right: at the bottom
    assert after_20[2] == approx(11.2402, abs=0.0001)  # too high: it jumps from wall to wall and never comes down


def test_3_epochs_with_24_then_1_then_6_mangoes_per_step(crate, crate_network, show):
    ends = []
    for batch_size in (24, 1, 6):
        crate_network.layers[0].neurons[0].bias = -2.2
        trainer = trainer_for(crate_network, GradientDescent(learning_rate=0.1))
        losses = trainer.train(crate, epochs=3, batch_size=batch_size, shuffle=Shuffle(random.Random(5)))
        steps = 3 * len(Batches(batch_size).cut(crate))
        show(batch_size, "per step ", steps, "steps   loss after each epoch", losses)
        ends.append((steps, round(losses[3], 4)))
    assert ends == [(3, 3.6846), (72, 1.001), (12, 1.0507)]


def test_10_epochs_in_batches_of_6_fall_fast_then_stay_flat(crate, crate_network, show):
    trainer = trainer_for(crate_network, GradientDescent(learning_rate=0.1))
    losses = trainer.train(crate, epochs=10, batch_size=6, shuffle=Shuffle(random.Random(5)))
    for epoch in range(len(losses)):
        show("after epoch", epoch, "  loss", losses[epoch])
    assert [round(loss, 4) for loss in losses[:5]] == [11.2402, 2.6045, 1.2986, 1.0507, 1.0064]
    for epoch in range(5, 11):
        assert losses[epoch] == approx(1.0002, abs=0.02)  # flat: more epochs change nothing


def test_every_weight_and_bias_gets_an_optimizer_of_its_own(mango_network, mango, show):
    trainer = trainer_for(mango_network, Momentum(learning_rate=0.1))
    slopes = trainer.batch_slopes.slopes(mango_network, [mango])
    trainer.step([mango])
    for n in range(3):
        own = trainer.optimizers[0][n]
        velocities = [optimizer.velocity for optimizer in own.weights]
        show("hidden neuron", n, " slopes", slopes[0][n].weights, " -> each weight's own velocity", velocities)
        assert velocities == approx([-0.1 * slope for slope in slopes[0][n].weights])
        assert own.bias.velocity == approx(-0.1 * slopes[0][n].bias)
    assert trainer.optimizers[0][0].weights[0] is not trainer.optimizers[0][0].weights[1]


def test_each_optimizer_trains_the_mango_network(new_mango_network, mango, show):
    ends = []
    for optimizer in (GradientDescent(0.5), Momentum(0.05), RMSProp(0.01), Adam(0.01)):
        network = new_mango_network()
        losses = trainer_for(network, optimizer).train([mango], epochs=40, batch_size=1,
                                                       shuffle=Shuffle(random.Random(5)))
        show(type(optimizer).__name__, " loss after 0, 10, 20, 40 epochs:", [losses[e] for e in (0, 10, 20, 40)],
             "  answer", network.answer(mango[0]))
        ends.append(losses[40])
        assert losses[40] < losses[20] < losses[10] < losses[0]
    assert ends[0] == approx(0.0014, abs=0.0001)
