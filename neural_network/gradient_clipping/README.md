# Gradient clipping

All the slopes of a network together are its gradient. Think of them as one
long arrow: how far to move every weight and bias. In a deep network the
arrow can become thousands or millions long, and one step then throws every
weight far away. Clipping puts a limit on the arrow's length.

`GradientClipping(limit)` has two methods:

| Method | What it does |
|--------|--------------|
| `length(slopes)` | Squares every slope, of every weight and bias in every layer, adds them up and takes the root |
| `clipped(slopes)` | Returns the slopes as they are when the length is within the limit. Otherwise it multiplies every slope by `limit / length`. |

Every slope is multiplied by the same number, so the direction of the arrow
stays the same. Only its size is capped.

`NO_LIMIT` is a ready made `GradientClipping` whose limit is infinity. It
never changes a slope. `Trainer` uses it unless it is given another.

```python
from neural_network.backpropagation import Backpropagation, NeuronSlopes
from neural_network.gradient_clipping import GradientClipping
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer

clipping = GradientClipping(limit=1.0)
slopes = [[NeuronSlopes(weights=[3600.0, 4800.0], bias=0.0)]]
clipping.length(slopes)                  # 6000.0
clipping.clipped(slopes)[0][0].weights   # [0.6, 0.8]

trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.05), clipping)
```

`tests/test_gradient_clipping.py` also measures the slopes of each layer in
ten sigmoid layers, where they vanish, and trains a network whose starting
weights are too big, with the limit and without it.

Uses: `backpropagation`. Used by: `training`.

Article: [Vanishing and exploding gradients](https://byteblueprints.com/artificial-intelligence/ann/17_gradients/).

## See it work

```bash
uv run pytest tests/test_gradient_clipping.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_gradient_clipping.py::test_slopes_of_3600_and_4800_are_6000_long_and_clip_to_0_6_and_0_8 PASSED
      slopes [3600.0, 4800.0]   length 6000.0
      shrink by limit / length = 1 / 6000  -> [0.6, 0.8]   length 1.0

tests/test_gradient_clipping.py::test_through_ten_sigmoid_layers_the_slopes_vanish PASSED
      slopes of layer 10 are 0.031 long
      slopes of layer 8 are 0.0019 long
      slopes of layer 6 are 8.8e-05 long
      slopes of layer 4 are 5.4e-06 long
      slopes of layer 2 are 3.2e-07 long
      slopes of layer 1 are 3.5e-08 long

tests/test_gradient_clipping.py::test_clipping_keeps_an_exploding_network_alive PASSED
      limit inf   loss after 0, 1, 2, 3 steps: ['7.313e+04', '9.78e+73', '2.302e+89', '2.302e+87']   after 3 steps: 2.302e+87
      limit 1.0   loss after 0, 1, 2, 3 steps: ['7.313e+04', '2.971e+04', '7150', '1051']   after 30 steps: 2566
```
