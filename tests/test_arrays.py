import random

import pytest
from pytest import approx

np = pytest.importorskip("numpy")

from neural_network.arrays import (ArrayBackpropagation, ArrayCrossEntropy, ArrayForwardPass, ArrayNetwork,
                                   ArrayTrainer, ExampleArrays)
from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.batches import BatchSlopes, Shuffle
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import CrossEntropy
from neural_network.training import Trainer

BACKPROPAGATION = ArrayBackpropagation(ArrayCrossEntropy())


def test_a_layer_holds_its_weights_with_one_column_for_each_neuron(mango_network, as_arrays, show):
    hidden, output = as_arrays(mango_network).layers
    show("hidden weights, row 1", hidden.weights[0].tolist(), "  row 2", hidden.weights[1].tolist())
    show("hidden biases", hidden.biases.tolist())
    assert hidden.weights.shape == (2, 3)
    assert hidden.weights[:, 0].tolist() == [0.8, -0.4]  # the first neuron's weights are the first column
    assert hidden.biases.tolist() == [0.1, -0.2, 0.3]
    assert output.weights.shape == (3, 1)
    assert output.weights[:, 0].tolist() == [1.2, -0.7, 0.5]


def test_the_examples_are_stacked_with_one_row_for_each_mango(labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    show("inputs", mangoes.inputs.shape, "  truths", mangoes.truths.shape)
    show("first mango", mangoes.inputs[0].tolist(), "  truth", mangoes.truths[0].tolist())
    assert (mangoes.inputs.shape, mangoes.truths.shape) == ((36, 2), (36, 1))
    assert mangoes.inputs[0].tolist() == [0.61, 0.66]
    assert mangoes.truths[0].tolist() == [1.0]


def test_one_step_gives_all_three_totals_for_four_mangoes(mango_network, as_arrays, labelled_crate, show):
    four = ExampleArrays(labelled_crate[:4])
    hidden = as_arrays(mango_network).layers[0]
    totals = hidden.totals(four.inputs)
    outputs = hidden.outputs(four.inputs)
    for e in range(4):
        show("mango", four.inputs[e].tolist(), "  totals", totals[e].tolist(), "  outputs", outputs[e].tolist())
    assert totals.shape == (4, 3)
    assert np.allclose(totals, [[0.324, 0.577, 0.264], [-0.172, 0.538, 0.664], [0.6, 0.481, 0.019], [0.24, 0.283, 0.277]])
    assert outputs[1].tolist() == approx([0.0, 0.538, 0.664])  # ReLU turns the total below 0 into 0
    # Each number is the total the plain neuron gives for that mango alone.
    for e in range(4):
        for n in range(3):
            assert totals[e, n] == approx(mango_network.layers[0].neurons[n].total(labelled_crate[e][0]))


def test_the_shape_of_every_array_on_the_way_forward(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    forward = ArrayForwardPass(as_arrays(mango_network), mangoes.inputs)
    hidden, output = forward.steps
    show("hidden: inputs", hidden.inputs.shape, " totals", hidden.totals.shape, " outputs", hidden.outputs.shape)
    show("output: inputs", output.inputs.shape, " totals", output.totals.shape, " outputs", output.outputs.shape)
    assert (hidden.inputs.shape, hidden.totals.shape, hidden.outputs.shape) == ((36, 2), (36, 3), (36, 3))
    assert (output.inputs.shape, output.totals.shape, output.outputs.shape) == ((36, 3), (36, 1), (36, 1))
    assert forward.answer.shape == (36, 1)


def test_every_answer_is_the_plain_networks_answer(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    answers = as_arrays(mango_network).answer(mangoes.inputs)
    show("first four answers", answers[:4, 0].tolist())
    assert answers[:4, 0].tolist() == approx([0.5042, 0.4639, 0.5727, 0.5321], abs=0.00005)
    for e in range(36):
        assert answers[e, 0] == approx(mango_network.answer(labelled_crate[e][0])[0], abs=1e-12)
    one = np.array([[1.0, 0.5]])  # the mango that is fully yellow and half soft
    assert as_arrays(mango_network).answer(one)[0, 0] == approx(0.5878, abs=0.00005)


def test_every_slope_array_has_the_shape_of_the_array_it_moves(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    network = as_arrays(mango_network)
    hidden, output = BACKPROPAGATION.slopes(network, mangoes.inputs, mangoes.truths)
    show("hidden weight slopes", hidden.weights.shape, "  bias slopes", hidden.biases.shape)
    show("output weight slopes", output.weights.shape, "  bias slopes", output.biases.shape)
    assert hidden.weights.shape == network.layers[0].weights.shape == (2, 3)
    assert hidden.biases.shape == network.layers[0].biases.shape == (3,)
    assert output.weights.shape == network.layers[1].weights.shape == (3, 1)
    assert output.biases.shape == network.layers[1].biases.shape == (1,)


def test_the_slopes_are_the_plain_average_slopes_over_the_36_mangoes(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    hidden, output = BACKPROPAGATION.slopes(as_arrays(mango_network), mangoes.inputs, mangoes.truths)
    plain = BatchSlopes(Backpropagation(CrossEntropy())).slopes(mango_network, labelled_crate)
    show("hidden weight slopes, row 1", hidden.weights[0].tolist(), "  row 2", hidden.weights[1].tolist())
    show("hidden bias slopes", hidden.biases.tolist())
    show("output weight slopes", output.weights[:, 0].tolist(), "  bias slope", output.biases.tolist())
    assert hidden.weights[0].tolist() == approx([-0.0969, 0.0541, -0.0292], abs=0.00005)
    assert output.weights[:, 0].tolist() == approx([-0.0363, -0.0553, 0.0298], abs=0.00005)
    for n in range(3):
        assert hidden.weights[:, n].tolist() == approx(plain[0][n].weights, abs=1e-12)
        assert hidden.biases[n] == approx(plain[0][n].bias, abs=1e-12)
    assert output.weights[:, 0].tolist() == approx(plain[1][0].weights, abs=1e-12)
    assert output.biases[0] == approx(plain[1][0].bias, abs=1e-12)


def test_the_slopes_match_a_tiny_nudge_of_each_number(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    hidden, output = BACKPROPAGATION.slopes(as_arrays(mango_network), mangoes.inputs, mangoes.truths)
    # The slow way: nudge one number, for one mango at a time, then take the average over the mangoes.
    nudged = [FiniteDifference(CrossEntropy()).slopes(mango_network, inputs, truths) for inputs, truths in labelled_crate]
    first_weight = sum(slopes[0][0].weights[0] for slopes in nudged) / 36
    output_bias = sum(slopes[1][0].bias for slopes in nudged) / 36
    show("first hidden weight: sweep", float(hidden.weights[0, 0]), "  nudge", first_weight)
    show("output bias:         sweep", float(output.biases[0]), "  nudge", output_bias)
    assert hidden.weights[0, 0] == approx(first_weight, abs=1e-5)
    assert output.biases[0] == approx(output_bias, abs=1e-5)


def right_answers(network: ArrayNetwork, mangoes: ExampleArrays) -> int:
    says_ripe = network.answer(mangoes.inputs) > 0.5
    return int((says_ripe == (mangoes.truths == 1)).sum())


def test_200_epochs_bring_the_loss_from_0_70_to_0_33(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    network = as_arrays(mango_network)
    right_before = right_answers(network, mangoes)
    trainer = ArrayTrainer(network, BACKPROPAGATION, GradientDescent(learning_rate=0.5))
    losses = trainer.train(mangoes.inputs, mangoes.truths, epochs=200)
    for e in (0, 1, 10, 50, 100, 200):
        show("after epoch", e, "  loss", losses[e])
    show("mangoes it gets right: before", right_before, "  after", right_answers(network, mangoes), "  of 36")
    assert [round(losses[e], 4) for e in (0, 1, 10, 50, 100, 200)] == [0.6966, 0.6802, 0.6159, 0.4531, 0.3691, 0.3323]
    assert (right_before, right_answers(network, mangoes)) == (19, 28)


def test_training_ends_on_the_plain_trainers_numbers(mango_network, new_mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    network = as_arrays(mango_network)
    losses = ArrayTrainer(network, BACKPROPAGATION, GradientDescent(0.5)).train(mangoes.inputs, mangoes.truths, 200)
    # The plain trainer, with all 36 mangoes in every batch, takes the same 200 steps one number at a time.
    plain = new_mango_network()
    plain_trainer = Trainer(plain, Backpropagation(CrossEntropy()), GradientDescent(0.5))
    plain_losses = plain_trainer.train(labelled_crate, epochs=200, batch_size=36, shuffle=Shuffle(random.Random(1)))
    first = plain.layers[0].neurons[0]
    show("loss after 200 epochs: arrays", losses[200], "  plain", plain_losses[200])
    show("first neuron's weights: arrays", network.layers[0].weights[:, 0].tolist(), "  plain", first.weights)
    assert losses == approx(plain_losses, abs=1e-9)
    assert network.layers[0].weights[:, 0].tolist() == approx(first.weights, abs=1e-9)
    assert network.layers[0].biases[0] == approx(first.bias, abs=1e-9)
