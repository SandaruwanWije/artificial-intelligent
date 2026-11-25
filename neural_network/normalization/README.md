# Normalization

In a deep network the numbers can shrink or grow from layer to layer. A
careful start sets their size once. A normalization step sets it again every
time the numbers pass through: whatever size comes in, the same size goes
out.

Both kinds here do the same four things to a group of numbers:

1. **Subtract the mean.** `mean(numbers)` is their average.
2. **Divide by the spread.** `spread(numbers)` is a typical distance from the
   mean: the root of the average squared distance. `standardized(numbers)`
   does both, and the result has a mean of 0 and a spread of 1.
3. **Scale and shift.** Every neuron multiplies its own number by a scale,
   called gamma, and adds a shift, called beta.
4. **Bend.** The neuron's activation.

They differ in which numbers form a group.

| Class | The group | Needs a batch | When the network is used |
|-------|-----------|---------------|--------------------------|
| `LayerNorm(size, activation)` | One example, over the neurons of the layer | No | The same as in training |
| `BatchNorm(size, activation, keep=0.9)` | One neuron, over the examples of the batch | Yes | Uses the means and spreads it kept from training |

`Normalization` is their base class. It is a kind of `Layer` and holds what
they share: the neurons, `mean`, `spread` and `standardized`.

Each neuron in a norm is a plain `Neuron` with one weight and a bias. The
weight is the scale and starts at 1. The bias is the shift and starts at 0.
So `Backpropagation` finds their slopes like any weight and bias, and
`Trainer` gives each one its own optimizer. Nothing else has to know about
them.

The numbers that come in should be plain sums, so the layer before a norm
uses `NO_BEND`:

```python
import random

from neural_network.activations import NO_BEND, RELU, SIGMOID
from neural_network.initialization import He, Xavier
from neural_network.network import Network
from neural_network.normalization import BatchNorm, LayerNorm

numbers = random.Random(3)
network = Network([
    He(numbers).layer(inputs=2, neurons=16, activation=NO_BEND),   # the sums
    LayerNorm(size=16, activation=RELU),                           # standardize, scale, shift, bend
    He(numbers).layer(inputs=16, neurons=16, activation=NO_BEND),
    BatchNorm(size=16, activation=RELU),
    Xavier(numbers).layer(inputs=16, neurons=1, activation=SIGMOID),
])
```

## How a norm fits in

`Layer` has four methods that training asks. A plain layer answers the first
two simply and has no use for the last two.

| Method | A plain layer | `LayerNorm` | `BatchNorm` |
|--------|---------------|-------------|-------------|
| `inputs_of(n, inputs)`: what neuron `n` reads | All the inputs | Its own input, standardized over the layer | Its own input, standardized over the batch |
| `blame_on_inputs(inputs, blame_at_totals)`: how blame goes back | Along every weight | Through the scale, then through the standardizing | The same, with averages over the batch |
| `see_inputs(batch_inputs)`: every example's inputs, before any goes through | Nothing | Nothing | Measures each neuron's mean and spread, and adds a little of them to the kept ones |
| `see_blame(batch_inputs, batch_blame_at_totals)`: every example's blame, before any goes back | Nothing | Nothing | Measures the two averages its blame needs |

For the last two to work, a batch goes through the network one layer at a
time: `BatchForwardPass` going forward, and `Backpropagation.slopes_for_each`
coming back.

## Batch norm in training and in use

In training `BatchNorm` standardizes with the batch it is shown. Each time,
it also updates a kept mean and spread for every neuron: `keep` of the old
value plus the rest of the new one. `outputs`, which `Network.answer` calls,
standardizes with the kept ones, so one example alone gets a proper answer.
`Trainer.loss` uses `Network.answer`, so it is the loss in use.

Three things to know:

- A new `BatchNorm` has kept nothing yet (means of 0, spreads of 1), so in
  use it changes nothing until it has trained on some batches.
- `FiniteDifference` nudges through `Network.answer`, so on a network with
  batch norm it gives the slopes in use, not the slopes in training. The
  tests check the training slopes with a nudge of the whole batch's loss.
- `EarlyStopping` copies weights and biases, not the kept means and spreads.

`tests/test_normalization.py` covers `LayerNorm` and `tests/test_batch_norm.py`
covers `BatchNorm`. Both standardize numbers by hand, check every slope with
a nudge, the scales and shifts too, and show that ten normed layers give the
same signal for three sizes of start.

Uses: `network`, `neuron`, `activations`.

Article: [Batch norm and layer norm](https://byteblueprints.com/artificial-intelligence/ann/18_normalization/).

## See it work

```bash
uv run pytest tests/test_normalization.py tests/test_batch_norm.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_normalization.py::test_standardizing_subtracts_the_mean_and_divides_by_the_spread PASSED
      the numbers    [60.2, 41.6, 24.7, 57.1, 30.4, 48.0, 54.6, 44.9]   mean 45.1875   spread 11.8065
      minus the mean [15.0125, -3.5875, -20.4875, 11.9125, -14.7875, 2.8125, 9.4125, -0.2875]
      / the spread   [1.2715, -0.3039, -1.7353, 1.009, -1.2525, 0.2382, 0.7972, -0.0244]

tests/test_batch_norm.py::test_batch_norm_goes_down_a_column_and_layer_norm_goes_along_a_row PASSED
      neuron 1 over the six mangoes: [0.85, -0.52, 0.39, -3.25, -3.36, 0.83]   mean -0.8433  spread 1.799
      mango 1 over the eight neurons: [0.85, 0.34, 0.03, 0.41, -0.88, 0.82, -0.27, -0.21]   mean 0.1362  spread 0.5503

tests/test_batch_norm.py::test_with_batch_norm_the_signal_is_the_same_and_all_three_starts_learn PASSED
      half of He  signal after the tenth norm 0.7531   loss in use after 0, 5, 10 epochs: [0.25, 0.2288, 0.2001]   loss in training 0.1687
      He  signal after the tenth norm 0.7531   loss in use after 0, 5, 10 epochs: [0.2584, 0.2657, 0.2337]   loss in training 0.1767
      twice He  signal after the tenth norm 0.7531   loss in use after 0, 5, 10 epochs: [0.5, 0.206, 0.1626]   loss in training 0.1916
```
