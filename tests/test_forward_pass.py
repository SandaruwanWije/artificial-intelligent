from pytest import approx

from neural_network.forward_pass import BatchForwardPass, ForwardPass


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


def test_a_batch_goes_through_one_layer_at_a_time_and_every_example_keeps_its_own_pass(mango_network, show):
    batch_inputs = [[1.0, 0.5], [0.2, 0.9], [0.7, 0.3]]
    batch = BatchForwardPass(mango_network, batch_inputs)
    for e in range(3):
        alone = ForwardPass(mango_network, batch_inputs[e])
        show("mango", batch_inputs[e], " hidden outputs", batch.passes[e].steps[0].outputs, "  answer", batch.passes[e].answer)
        assert batch.passes[e].steps == alone.steps  # plain layers do not look at the rest of the batch
        assert batch.passes[e].answer == mango_network.answer(batch_inputs[e])
    assert len(batch.passes) == 3
