# Optimizers

Backpropagation gives every weight a slope. An optimizer is the rule that
turns that slope into a step. `Optimizer` is the base class. Every optimizer
has the same two methods:

| Method | What it does |
|--------|--------------|
| `step(weight, slope)` | Returns the weight after one step downhill |
| `fresh()` | Returns a new optimizer with the same settings and an empty memory |

An optimizer object belongs to one weight. Whatever it remembers, it
remembers about that weight only. So `step(weight, slope)` never has to be
told which weight it is moving. `Trainer` calls `fresh()` once for every
weight and bias, and keeps them in `NeuronOptimizers` (the `weights` and the
`bias` of one neuron).

| Class | What it remembers | Its step |
|-------|-------------------|----------|
| `GradientDescent(learning_rate)` | Nothing | Minus the learning rate times the slope. It lives in `gradient_descent`. |
| `Momentum(learning_rate, keep=0.9)` | `velocity`, its last step | 90% of the last step, plus the plain step |
| `RMSProp(learning_rate, keep=0.9)` | `average_square`, a running average of its slopes squared | The plain step, with the learning rate divided by the usual size of its slopes |
| `Adam(learning_rate, keep_slope=0.9, keep_square=0.999)` | `average_slope` and `average_square` | The learning rate times the average slope, divided by the usual size |

```python
from neural_network.optimizers import Momentum

for_w1 = Momentum(learning_rate=0.1)
for_w2 = for_w1.fresh()             # the same settings, a memory of its own
w1 = for_w1.step(-3.2, slope=-0.32)   # -3.168
w2 = for_w2.step(0.9, slope=5.4)      # 0.36
```

To train a network with one, hand it to the trainer:
`Trainer(network, Backpropagation(SquaredError()), Adam(learning_rate=0.01))`.

`tests/test_optimizers.py` runs all four on a ravine: a loss with two
weights, where the walls are 60 times steeper than the floor.
`tests/test_training.py` trains a network with each of them.

Uses: nothing. Used by: `gradient_descent`, `training`.

Article: [SGD, Momentum, RMSProp, Adam](https://byteblueprints.com/artificial-intelligence/ann/12_optimizers/).

## See it work

```bash
uv run pytest tests/test_optimizers.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_optimizers.py::test_momentum_speeds_up_along_the_floor_and_is_slow_to_turn_on_the_walls PASSED
      steps along w1: [0.032, 0.0605, 0.0855, 0.1072, 0.1256, 0.1409, 0.1533, 0.1629, 0.17, 0.1746]
      step 3 for w2: plain step 0.2052   kept -0.6318   new step -0.4266
      it rolls past the bottom to w1 = 0.4901   after 60 steps: 0.0265

tests/test_optimizers.py::test_all_four_on_the_ravine_for_60_steps PASSED
      GradientDescent  close after 55 steps   distance after 60: 0.5146
      Momentum  close after 20 steps   distance after 60: 0.0265
      RMSProp  close after 25 steps   distance after 60: 0.0001
      Adam  close after 30 steps   distance after 60: 0.156
```
