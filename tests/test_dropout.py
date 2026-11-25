import random

from pytest import approx

from neural_network.backpropagation import Backpropagation, FiniteDifference
from neural_network.dropout import NO_DROPOUT, Dropout
from neural_network.forward_pass import ForwardPass
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer

SEED = 10  # with this seed the first pass switches off the middle hidden neuron


def test_a_switch_is_0_for_about_half_the_neurons_and_2_for_the_rest(show):
    dropout = Dropout(share=0.5, random=random.Random(SEED))
    switches = [dropout.switch() for _ in range(1000)]
    show("the first ten switches:", switches[:10])
    show("of 1000 neurons,", switches.count(0.0), "are off and", switches.count(2.0), "are on and count double")
    assert switches.count(0.0) + switches.count(2.0) == 1000
    assert switches.count(0.0) == approx(500, abs=50)
    assert [NO_DROPOUT.switch() for _ in range(5)] == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_the_forward_pass_keeps_which_neurons_were_off(mango_network, mango, show):
    inputs, _ = mango
    plain = ForwardPass(mango_network, inputs)
    dropped = ForwardPass(mango_network, inputs, Dropout(share=0.5, random=random.Random(SEED)))
    hidden, output = dropped.steps
    show("with every neuron on: hidden outputs", plain.steps[0].outputs, "  answer", plain.answer)
    show("switches", hidden.switches, "  hidden outputs", hidden.outputs, "  answer", dropped.answer)
    assert hidden.switches == [2.0, 0.0, 2.0]
    assert hidden.outputs == approx([0.7 * 2, 0.0, 0.0])
    assert output.switches == [1.0]  # the last layer gives the answer, so it is never switched off
    assert plain.steps[0].switches == [1.0, 1.0, 1.0]


def test_backpropagation_skips_the_same_neurons(mango_network, mango, show):
    backpropagation = Backpropagation(SquaredError(), Dropout(share=0.5, random=random.Random(SEED)))
    slopes = backpropagation.slopes(mango_network, *mango)
    for n in range(3):
        show("hidden neuron", n, " weight slopes", slopes[0][n].weights, "  bias slope", slopes[0][n].bias)
    show("the answer neuron's weight slopes", slopes[1][0].weights)
    assert (slopes[0][1].weights, slopes[0][1].bias) == ([0.0, 0.0], 0.0)  # switched off: no blame comes back
    assert slopes[0][0].weights[0] != 0.0
    assert slopes[1][0].weights[1] == 0.0  # and its wire to the answer carried nothing


def test_every_slope_with_dropout_matches_a_nudge_with_the_same_neurons_off(mango_network, mango, show):
    inputs, truths = mango
    loss, nudge = SquaredError(), FiniteDifference(SquaredError())
    slopes = Backpropagation(loss, Dropout(0.5, random.Random(SEED))).slopes(mango_network, inputs, truths)

    def loss_now() -> float:  # the same seed switches off the same neurons every time
        return loss.total(ForwardPass(mango_network, inputs, Dropout(0.5, random.Random(SEED))).answer, truths)

    for layer_number in range(len(mango_network.layers)):
        for n in range(len(mango_network.layers[layer_number].neurons)):
            neuron = mango_network.layers[layer_number].neurons[n]
            nudged = []
            for i in range(len(neuron.weights)):
                before = loss_now()
                neuron.weights[i] = neuron.weights[i] + nudge.nudge
                nudged.append(nudge.slope(before, loss_now()))
                neuron.weights[i] = neuron.weights[i] - nudge.nudge
            show("layer", layer_number, "neuron", n, " swept", slopes[layer_number][n].weights, "  nudged", nudged)
            assert slopes[layer_number][n].weights == approx(nudged, abs=1e-5)


def test_dropout_is_only_for_training(mango_network, mango, show):
    trainer = Trainer(mango_network, Backpropagation(SquaredError(), Dropout(0.5, random.Random(SEED))),
                      GradientDescent(learning_rate=0.5))
    answers = [mango_network.answer(mango[0])[0] for _ in range(3)]
    losses = [trainer.loss([mango]) for _ in range(3)]
    show("asked three times, the network answers", answers, "  and the trainer's loss is", losses)
    assert answers == approx([0.5878, 0.5878, 0.5878], abs=0.0001)  # every neuron is on
    assert losses[0] == losses[1] == losses[2]


def test_over_many_passes_a_layer_adds_up_to_about_the_same_as_with_no_dropout(mango_network, mango, show):
    dropout, total = Dropout(share=0.5, random=random.Random(SEED)), 0.0
    for _ in range(4000):
        outputs = ForwardPass(mango_network, mango[0], dropout).steps[0].outputs
        total = total + outputs[0] + outputs[1] + outputs[2]
    show("every neuron on: the hidden outputs add up to", 0.7 + 0.55 + 0.0)
    show("half of them off, the rest doubled: on average over 4000 passes", total / 4000)
    assert total / 4000 == approx(1.25, rel=0.05)


def test_a_step_with_dropout_leaves_the_switched_off_neuron_where_it_was(mango_network, mango, show):
    trainer = Trainer(mango_network, Backpropagation(SquaredError(), Dropout(0.5, random.Random(SEED))),
                      GradientDescent(learning_rate=0.5))
    trainer.step([mango])
    for neuron in mango_network.layers[0].neurons:
        show("after one step: weights", neuron.weights, "  bias", neuron.bias)
    assert mango_network.layers[0].neurons[1].weights == [0.3, 0.9]  # it was off in this pass, so it did not move
    assert mango_network.layers[0].neurons[0].weights != [0.8, -0.4]
