# Batches

Training on many examples needs two choices before it starts. The batch size
is how many examples the network reads before each step. The number of epochs
is how many times it goes through all the examples. Three small classes do the
work for one epoch.

| Class | What it does |
|-------|--------------|
| `Shuffle(random)` | `shuffled(examples)` returns the same examples in a new random order. The list it was given keeps its order. |
| `Batches(size)` | `cut(examples)` cuts a list into batches of `size`. What is left over makes one last, smaller batch. |
| `BatchSlopes(backpropagation)` | `slopes(network, batch)` gets each example's slopes from `Backpropagation.slopes_for_each`, then averages them. `average` does it for one neuron. |

`Shuffle` takes a `random.Random(seed)`, so the same seed gives the same
order every run. `BatchSlopes` returns the same shape as `Backpropagation`:
one list per layer, one `NeuronSlopes` per neuron.

```python
import random

from neural_network.backpropagation import Backpropagation
from neural_network.batches import Batches, BatchSlopes, Shuffle
from neural_network.loss import SquaredError

mixed = Shuffle(random.Random(5)).shuffled(examples)   # a new order
batches = Batches(6).cut(mixed)                        # 24 examples -> 4 batches of 6
slopes = BatchSlopes(Backpropagation(SquaredError())).slopes(network, batches[0])
```

A batch of all the examples gives a steady slope but only one step per epoch.
A batch of one gives many steps, but each one follows a single example. A
small batch is in between.

`tests/test_batches.py` shuffles and cuts a crate of 24 mangoes, works out a
batch slope by hand, and checks the averaged slopes against `FiniteDifference`.
`tests/test_training.py` trains on the crate with different batch sizes.

Uses: `backpropagation`, `network`. Used by: `training`, `splits`.

Article: [Learning rate, epochs, batch size](https://byteblueprints.com/artificial-intelligence/ann/11_hyperparameters/).

## See it work

```bash
uv run pytest tests/test_batches.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_batches.py::test_a_batch_slope_is_the_average_of_its_mangoes_slopes PASSED
      the number -2.2   this mango wants 2.62   its slope 2 x miss = -9.64
      the number -2.2   this mango wants 2.03   its slope 2 x miss = -8.46
      the number -2.2   this mango wants -0.71   its slope 2 x miss = -2.98
      the number -2.2   this mango wants 1.65   its slope 2 x miss = -7.7
      the number -2.2   this mango wants 0.26   its slope 2 x miss = -4.92
      the number -2.2   this mango wants 1.61   its slope 2 x miss = -7.62
      total -41.32   / 6 mangoes = batch slope -6.8867
```
