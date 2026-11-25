# Gradient descent

`GradientDescent(learning_rate)` has one move, `step(weight, slope)`. It moves
the weight against its slope, by the learning rate:

```
new weight = weight - learning rate * slope
```

When the slope is negative, the loss goes down to the right, so the weight
moves right. When it is positive, the weight moves left. Steep slopes make big
steps and gentle slopes make small ones, so the steps shrink near the bottom.

```python
from neural_network.gradient_descent import GradientDescent

descent = GradientDescent(learning_rate=0.8)
descent.step(-2.2, slope=-2.24)   # -0.408
```

A learning rate that is too big jumps past the bottom and climbs out.
`tests/test_gradient_descent.py` shows both.

`GradientDescent` is the simplest `Optimizer`. It remembers nothing, so
`fresh()` returns another one with the same learning rate. The optimizers
that do remember are in `optimizers`.

Uses: `optimizers`. Used by: `training`.

Article: [Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/).

## See it work

```bash
uv run pytest tests/test_gradient_descent.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_gradient_descent.py::test_steps_walk_down_to_the_lowest_point PASSED
      start   weight -2.2 loss 3.884
      step 1   weight -0.408 loss 0.9939
      step 2   weight 0.3805 loss 0.4343
      step 3   weight 0.7274 loss 0.326
      step 5   weight 0.9472 loss 0.301
      step 10   weight 0.9991 loss 0.3
      step 30   weight 1.0 loss 0.3
```
