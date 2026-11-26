from importlib.util import find_spec

import pytest
from pytest import approx

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
# Keras is not loaded here: the library picks what Keras runs on before it loads it.
if find_spec("keras") is None:
    pytest.skip("Keras is not installed", allow_module_level=True)

from neural_network.arrays import ArrayBackpropagation, ArrayCrossEntropy, ArrayNetwork, ArrayTrainer, ExampleArrays
from neural_network.gradient_descent import GradientDescent
from neural_network.libraries import KerasNetwork, TorchNetwork, TorchSlopes, TorchTrainer

BACKPROPAGATION = ArrayBackpropagation(ArrayCrossEntropy())


def torch_copy(arrays: ArrayNetwork) -> TorchNetwork:
    copy = TorchNetwork(inputs=2, hidden=3)
    copy.start_from(arrays)
    return copy


def keras_copy(arrays: ArrayNetwork) -> KerasNetwork:
    copy = KerasNetwork(inputs=2, hidden=3, learning_rate=0.5)
    copy.start_from(arrays)
    return copy


def test_pytorch_keeps_one_row_for_each_neuron(mango_network, as_arrays, show):
    network = torch_copy(as_arrays(mango_network))
    show("hidden weight", tuple(network.hidden.weight.shape), "  first row", network.hidden.weight[0].tolist())
    show("output weight", tuple(network.output.weight.shape), "  bias", network.output.bias.tolist())
    assert tuple(network.hidden.weight.shape) == (3, 2)  # the arrays keep these weights as (2, 3)
    assert network.hidden.weight[0].tolist() == [0.8, -0.4]  # the first neuron's weights are the first row
    assert network.hidden.bias.tolist() == [0.1, -0.2, 0.3]
    assert tuple(network.output.weight.shape) == (1, 3)
    assert network.output.weight[0].tolist() == [1.2, -0.7, 0.5]


def test_pytorch_gives_the_arrays_answers(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    ours = as_arrays(mango_network).answer(mangoes.inputs)
    with torch.no_grad():
        theirs = torch_copy(as_arrays(mango_network))(torch.tensor(mangoes.inputs)).numpy()
    show("first four answers: arrays", ours[:4, 0].tolist(), "  PyTorch", theirs[:4, 0].tolist())
    assert theirs.shape == (36, 1)
    assert theirs[:4, 0].tolist() == approx([0.5042, 0.4639, 0.5727, 0.5321], abs=0.00005)
    assert np.allclose(theirs, ours, rtol=0, atol=1e-12)


def test_the_slopes_pytorch_finds_are_the_slopes_worked_out_by_hand(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    arrays = as_arrays(mango_network)
    by_hand = BACKPROPAGATION.slopes(arrays, mangoes.inputs, mangoes.truths)
    found = TorchSlopes(torch.nn.BCELoss()).slopes(torch_copy(arrays), mangoes.inputs, mangoes.truths)
    show("hidden weight slopes, row 1: by hand", by_hand[0].weights[0].tolist(), "  PyTorch", found[0].weights[0].tolist())
    show("output weight slopes:        by hand", by_hand[1].weights[:, 0].tolist(), "  PyTorch", found[1].weights[:, 0].tolist())
    assert found[0].weights[0].tolist() == approx([-0.0969, 0.0541, -0.0292], abs=0.00005)
    for layer_number in range(2):
        assert found[layer_number].weights.shape == by_hand[layer_number].weights.shape
        assert np.allclose(found[layer_number].weights, by_hand[layer_number].weights, rtol=0, atol=1e-12)
        assert np.allclose(found[layer_number].biases, by_hand[layer_number].biases, rtol=0, atol=1e-12)


def test_200_epochs_in_pytorch_follow_the_arrays_loss_at_every_step(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    arrays = as_arrays(mango_network)
    network = torch_copy(arrays)
    ours = ArrayTrainer(arrays, BACKPROPAGATION, GradientDescent(0.5)).train(mangoes.inputs, mangoes.truths, 200)
    theirs = TorchTrainer(network, learning_rate=0.5).train(mangoes.inputs, mangoes.truths, epochs=200)
    furthest = max(abs(ours[e] - theirs[e]) for e in range(201))
    for e in (0, 1, 50, 100, 200):
        show("after epoch", e, "  arrays", ours[e], "  PyTorch", theirs[e])
    show("the furthest the two losses are ever apart", f"{furthest:.1e}")
    assert len(theirs) == 201
    assert [round(theirs[e], 4) for e in (0, 1, 50, 100, 200)] == [0.6966, 0.6802, 0.4531, 0.3691, 0.3323]
    assert theirs == approx(ours, abs=1e-9)
    assert network.hidden.weight[0].tolist() == approx(arrays.layers[0].weights[:, 0].tolist(), abs=1e-9)


def test_keras_keeps_one_column_for_each_neuron_and_gives_the_same_answers(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    arrays = as_arrays(mango_network)
    network = keras_copy(arrays)
    kernel, bias = network.model.layers[0].get_weights()
    answers = network.answer(mangoes.inputs)
    show("hidden weights", kernel.shape, "  first column", kernel[:, 0].tolist(), "  biases", bias.tolist())
    show("first four answers", answers[:4, 0].tolist())
    assert kernel.shape == (2, 3)  # the same layout as the arrays
    # Keras keeps 32 bit numbers, so 0.8 is held as 0.800000012.
    assert kernel[:, 0].tolist() == approx([0.8, -0.4], abs=1e-7)
    assert answers[:4, 0].tolist() == approx([0.5042, 0.4639, 0.5727, 0.5321], abs=0.00005)
    assert np.allclose(answers, arrays.answer(mangoes.inputs), rtol=0, atol=1e-6)


def test_one_call_to_fit_trains_keras_like_the_arrays(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    arrays = as_arrays(mango_network)
    network = keras_copy(arrays)
    ours = ArrayTrainer(arrays, BACKPROPAGATION, GradientDescent(0.5)).train(mangoes.inputs, mangoes.truths, 200)
    theirs = network.train(mangoes.inputs, mangoes.truths, epochs=200)
    for e in (0, 1, 50, 100, 200):
        show("after epoch", e, "  arrays", ours[e], "  Keras", theirs[e])
    assert len(theirs) == 201
    assert [round(theirs[e], 4) for e in (0, 1, 50, 100, 200)] == [0.6966, 0.6802, 0.4531, 0.3691, 0.3323]
    assert theirs == approx(ours, abs=1e-5)
