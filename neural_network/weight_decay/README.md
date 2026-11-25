# Weight decay

A network that memorises its examples needs some large weights to do it.
Weight decay makes large weights cost something. It adds a second part to
the loss: a penalty that grows with every weight squared. To lower the whole
loss, training keeps a weight large only when the data really needs it.

`WeightDecay(strength)` has two methods. `strength` is the number often
called lambda: how expensive large weights are.

| Method | What it does |
|--------|--------------|
| `penalty(network)` | What the weights cost: half the strength times every weight squared, all added up. Biases are free. |
| `added(network, slopes)` | The slopes with the penalty's slope added. For each weight that is the strength times the weight. Bias slopes stay as they were. |

So at every step a weight first shrinks a little toward 0, by the learning
rate times the strength times the weight, and then takes its usual step.
A weight that the data does not push back decays toward 0.

In a `LayerNorm` layer a neuron's weight is its scale, so weight decay pulls
the scales toward 0 too.

`NO_DECAY` is a ready made `WeightDecay(0.0)`. It adds nothing. `Trainer`
uses it unless it is given another.

```python
from neural_network.backpropagation import Backpropagation
from neural_network.gradient_clipping import NO_LIMIT
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer
from neural_network.weight_decay import WeightDecay

trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5),
                  NO_LIMIT, WeightDecay(strength=0.003))
```

`tests/test_weight_decay.py` works out the penalty and one step by hand,
checks the added slope with a nudge, and trains a network of 20 neurons with
no decay, a little, and far too much.

Uses: `backpropagation`, `network`. Used by: `training`.

Article: [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/).

## See it work

```bash
uv run pytest tests/test_weight_decay.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_weight_decay.py::test_a_weight_first_shrinks_then_takes_its_usual_step PASSED
      weight 0.8   shrink 0.5 x 0.3 x 0.8 = 0.12   usual step 0.5 x slope = -0.1198
      new weight = 0.8 - 0.12 - -0.1198 = 0.7998

tests/test_weight_decay.py::test_more_decay_means_smaller_weights_and_too_much_crushes_them PASSED
      strength 0.0   squared weights 65.2046   training loss 0.0829   validation loss 0.1974
      strength 0.003   squared weights 29.8095   training loss 0.0959   validation loss 0.1785
      strength 0.3   squared weights 0.0   training loss 0.2483   validation loss 0.2591
```
