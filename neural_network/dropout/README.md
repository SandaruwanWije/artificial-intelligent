# Dropout

During training, dropout switches off a random part of the hidden neurons at
every pass. A neuron that may vanish at any moment cannot be the only one
that knows something, so what the network learns has to be spread over many
neurons.

`Dropout(share, random)` has one method:

| Method | What it gives |
|--------|---------------|
| `switch()` | The number one neuron's output is multiplied by in this pass. With a chance of `share` it is 0: the neuron is off. Otherwise it is `1 / (1 - share)`, so the neurons left on count for more and the layer adds up to about the same. |

The other pieces use it like this:

- `ForwardPass(network, inputs, dropout)` asks for one switch per hidden
  neuron, multiplies the neuron's output by it, and keeps it in the layer's
  `LayerStep`, in `switches`. The last layer gives the answer, so it is
  never switched off.
- `Backpropagation(loss, dropout)` runs that forward pass. On the way back it
  multiplies each neuron's blame by the switch that the step kept. So the
  neurons that were off going forward are the same ones that are off going
  back, and they get slopes of 0.

Dropout is only for training. `Network.answer` and `Trainer.loss` never use
it, so every neuron is on when the network is used or graded.

`NO_DROPOUT` is a ready made `Dropout` with a share of 0. Its switch is
always 1. `ForwardPass` and `Backpropagation` use it unless they are given
another.

```python
import random

from neural_network.backpropagation import Backpropagation
from neural_network.dropout import Dropout
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer

dropout = Dropout(share=0.5, random=random.Random(10))
trainer = Trainer(network, Backpropagation(SquaredError(), dropout), GradientDescent(learning_rate=0.5))
```

`tests/test_dropout.py` shows the switches in a forward pass, checks that
the backward pass skips the same neurons, and checks every slope against a
nudge with the same neurons off.

Uses: nothing. Used by: `forward_pass`, `backpropagation`.

Article: [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/).

## See it work

```bash
uv run pytest tests/test_dropout.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_dropout.py::test_the_forward_pass_keeps_which_neurons_were_off PASSED
      with every neuron on: hidden outputs [0.7, 0.55, 0.0]   answer [0.5878]
      switches [2.0, 0.0, 2.0]   hidden outputs [1.4, 0.0, 0.0]   answer [0.8292]

tests/test_dropout.py::test_backpropagation_skips_the_same_neurons PASSED
      hidden neuron 0  weight slopes [-0.1161, -0.0581]   bias slope -0.1161
      hidden neuron 1  weight slopes [0.0, 0.0]   bias slope 0.0
      hidden neuron 2  weight slopes [0.0, 0.0]   bias slope 0.0
      the answer neuron's weight slopes [-0.0677, 0.0, 0.0]
```
