# Backpropagation

To learn, a network needs a slope for every weight and bias: how the loss
moves when only that one number moves. Both classes here return them in the
same shape, one list per layer with one `NeuronSlopes` (the `weights` and the
`bias`) per neuron, in the same order as the network.

## The slow way: `FiniteDifference`

`FiniteDifference(loss, nudge).slopes(network, inputs, truths)` nudges one
number at a time, runs the network again with `loss_of`, and divides the
change in loss by the nudge. Every other number is held still, and each one is
put back before the next is nudged. It is easy to follow, but it needs one
more run of the network for every number.

## The fast way: `Backpropagation`

`Backpropagation(loss).slopes(network, inputs, truths)` finds the same slopes
in one sweep backwards.

1. Run a forward pass and keep every layer's step.
2. Start at the loss: `loss.slope` says how the loss moves with each answer.
   This is the blame.
3. For each layer, from the last one back, `back_to_totals` does this:
   - Multiply the blame by the activation's slope at each neuron's total.
     That gives the blame at the total.
   - A weight's slope is the blame at its neuron's total times the input on
     its wire. A bias's slope is the blame at the total itself.
   - Share the blame back to the layer before: each input gets the blame at
     every total it feeds, times the weight on that wire. The layer does
     this itself, in `Layer.blame_on_inputs`.

`slopes` is for one example. It is a batch of one for `slopes_for_each`,
which takes any batch and returns every example's slopes. There all the
examples go forward one layer at a time (`BatchForwardPass`) and back one
layer at a time, so a layer such as `BatchNorm` can look at the whole batch.

`tests/test_backpropagation.py` checks every slope of the sweep against
`FiniteDifference`. `tests/test_finite_difference.py` nudges one weight by
0.01 and shows the loss move.

`Backpropagation(loss, dropout)` takes an optional `Dropout`. It hands it to
the forward pass, and on the way back it multiplies each neuron's blame by the
switch the step kept. So a neuron that was off going forward gets slopes of 0.

Uses: `forward_pass`, `loss`, `network`, `dropout`. Used by: `batches`, `gradient_clipping`, `weight_decay`, `training`.

Articles: [Backpropagation part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/) and [part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/).

## See it work

```bash
uv run pytest tests/test_finite_difference.py tests/test_backpropagation.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_backpropagation.py::test_the_blame_flows_back_one_layer_at_a_time PASSED
      at the answer: loss slope -0.8243
      back through the sigmoid: x its slope 0.2423 -> -0.1997
      back along weight 1.2 -> -0.2397   x ReLU slope 1.0 -> -0.2397
      back along weight -0.7 -> 0.1398   x ReLU slope 1.0 -> 0.1398
      back along weight 0.5 -> -0.0999   x ReLU slope 0.0 -> 0.0
```
