from pytest import approx

from neural_network.forward_pass import ForwardPass


def test_every_layer_step_is_kept(mango_network, mango, show):
    inputs, _ = mango
    hidden, output = ForwardPass(mango_network, inputs).steps
    for name, step in (("hidden", hidden), ("output", output)):
        show(name, "layer got", step.inputs, " totals", step.totals, " outputs", step.outputs)
    assert hidden.inputs == [1.0, 0.5]
    assert hidden.totals == approx([0.7, 0.55, -0.05])
    assert hidden.outputs == approx([0.7, 0.55, 0])
    assert output.inputs == hidden.outputs
    assert output.totals == approx([0.355])


def test_it_gives_the_same_answer_as_the_network(mango_network, mango, show):
    inputs, _ = mango
    forward = ForwardPass(mango_network, inputs)
    show("forward pass answer", forward.answer, "  network answer", mango_network.answer(inputs))
    assert forward.answer == mango_network.answer(inputs)
