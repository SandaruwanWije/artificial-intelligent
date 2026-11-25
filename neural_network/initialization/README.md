# Initialization

Before the first training step, every weight needs a starting value. Two
things must be right. The weights must be different from each other, or the
neurons of a layer stay identical for ever. And they must be the right size,
or the signal fades or blows up on its way through a deep network.

`Initialization(random)` is the base class. It builds layers of new neurons:

| Method | What it does |
|--------|--------------|
| `spread(inputs)` | How far from 0 the weights usually lie, for a neuron with that many inputs. Each rule gives its own. |
| `layer(inputs, neurons, activation)` | A `Layer` of new neurons. Each weight is a random number around 0 with the rule's spread. Each bias is 0. |

| Class | Spread | Use it with |
|-------|--------|-------------|
| `Xavier` | 1 / √n | tanh and sigmoid |
| `He` | √2 / √n | ReLU |

Here n is the number of inputs. A neuron adds up n numbers, and that sum is
about √n times as big as one of them. Dividing the weights by √n brings it
back to the size of one input. ReLU switches off about half of the neurons,
so `He` makes the weights √2 times bigger to make up for it.

It takes a `random.Random(seed)`, so the same seed gives the same weights
every run.

```python
import random

from neural_network.activations import RELU, SIGMOID
from neural_network.initialization import He, Xavier
from neural_network.network import Network

numbers = random.Random(3)
network = Network([
    He(numbers).layer(inputs=2, neurons=16, activation=RELU),
    He(numbers).layer(inputs=16, neurons=16, activation=RELU),
    Xavier(numbers).layer(inputs=16, neurons=1, activation=SIGMOID),
])
```

`tests/test_initialization.py` measures the spread of 4000 weights, follows
the signal through ten layers of 16 neurons for four sizes of start, and
trains networks from the same start, from zero and from a random start.

Uses: `activations`, `neuron`, `network`.

Article: [Weight initialization](https://byteblueprints.com/artificial-intelligence/ann/16_init/).

## See it work

```bash
uv run pytest tests/test_initialization.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_initialization.py::test_only_the_right_spread_keeps_the_signal_steady_through_ten_layers PASSED
      half of He  spread 0.1768   signal at layers 1, 4, 7, 10: [0.2139, 0.0116, 0.0026, 0.0004]
      He  spread 0.3536   signal at layers 1, 4, 7, 10: [0.4278, 0.186, 0.3318, 0.4095]
      twice He  spread 0.7071   signal at layers 1, 4, 7, 10: [0.8556, 2.9753, 42.472, 419.3611]
      Xavier  spread 0.25   signal at layers 1, 4, 7, 10: [0.3025, 0.0465, 0.0293, 0.0128]

tests/test_initialization.py::test_neurons_that_start_the_same_stay_the_same PASSED
      after 300 epochs: weights [1.8347, 1.7154]   bias -1.2804
      after 300 epochs: weights [1.8347, 1.7154]   bias -1.2804
      after 300 epochs: weights [1.8347, 1.7154]   bias -1.2804
      loss 0.3833 -> 0.1145
```
