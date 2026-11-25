from pytest import approx

from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.forward_pass import ForwardPass
from neural_network.loss import SquaredError

BACKPROPAGATION = Backpropagation(SquaredError())


def test_the_mango_starts_with_a_loss_of_0_17(mango_network, mango, show):
    inputs, truths = mango
    loss = FiniteDifference(SquaredError()).loss_of(mango_network, inputs, truths)
    show("answer", mango_network.answer(inputs), "truth", truths, "-> loss", loss)
    assert loss == approx(0.1699, abs=0.0001)


def test_the_blame_flows_back_one_layer_at_a_time(mango_network, mango, show):
    inputs, truths = mango
    hidden_step, output_step = ForwardPass(mango_network, inputs).steps
    hidden, output = mango_network.layers
    blame = BACKPROPAGATION.loss.slope(output_step.outputs[0], truths[0])
    show("at the answer: loss slope", blame)
    at_total = blame * output.neurons[0].activation.slope(output_step.totals[0])
    show("back through the sigmoid: x its slope", output.neurons[0].activation.slope(output_step.totals[0]), "->", at_total)
    expected = []
    for neuron, weight, z in zip(hidden.neurons, output.neurons[0].weights, hidden_step.totals):
        shared = at_total * weight
        expected.append(shared * neuron.activation.slope(z))
        show("back along weight", weight, "->", shared, "  x ReLU slope", neuron.activation.slope(z), "->", expected[-1])
    hidden_slopes, output_slopes = BACKPROPAGATION.slopes(mango_network, inputs, truths)
    assert output_slopes[0].bias == approx(at_total)
    assert [neuron.bias for neuron in hidden_slopes] == approx(expected)


def test_the_slope_of_every_weight_and_bias(mango_network, mango, show):
    hidden, output = BACKPROPAGATION.slopes(mango_network, *mango)
    for number, neuron in enumerate(hidden, start=1):
        show("hidden neuron", number, "  weight slopes", neuron.weights, "  bias slope", neuron.bias)
    show("output neuron    weight slopes", output[0].weights, "  bias slope", output[0].bias)
    assert [[round(s, 2) for s in neuron.weights] for neuron in hidden] == [
        [-0.24, -0.12], [0.14, 0.07], [0, 0]]
    assert [round(neuron.bias, 2) for neuron in hidden] == [-0.24, 0.14, 0]
    assert [round(s, 2) for s in output[0].weights] == [-0.14, -0.11, 0]
    assert round(output[0].bias, 2) == -0.20


def test_the_sweep_agrees_with_nudging_each_number_one_at_a_time(mango_network, mango, show):
    swept = BACKPROPAGATION.slopes(mango_network, *mango)
    nudged = FiniteDifference(SquaredError()).slopes(mango_network, *mango)
    for layer_number in range(len(swept)):
        for n in range(len(swept[layer_number])):
            sweep, nudge = swept[layer_number][n], nudged[layer_number][n]
            show("layer", layer_number + 1, "neuron", n + 1, "  sweep", sweep.weights + [sweep.bias],
                 "  nudge", nudge.weights + [nudge.bias])
            assert sweep.weights == approx(nudge.weights, abs=1e-4)
            assert sweep.bias == approx(nudge.bias, abs=1e-4)
