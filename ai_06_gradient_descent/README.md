# AI #06: Gradient descent

Companion code for the article
[Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/).
It uses the article's words and numbers, so you can read the two side by side.

Training cannot see the whole loss. It feels the slope under the weight and
takes a step the other way, scaled by the learning rate:

```
slope  = how much the loss would change if the weight grew by 1
weight <- weight - learning_rate * slope
```

With two weights, each weight gets its own slope. Together they are the
gradient, and every weight steps against its own slope at once.

The one-weight valley starts at -2.2 and steps with a learning rate of 0.8:
the steps go 1.79, 0.79, 0.35, 0.15, 0.07, smaller every time. The long,
narrow bowl starts at w1 -3.2, w2 0.9 (loss 7.55) and is tried with three
learning rates: 0.03 crawls, 0.15 reaches the bottom, 0.34 flies out.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_06_gradient_descent
uv run pytest ai_06_gradient_descent
```

It prints each chapter of the article and saves five pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `slope.png` | The slope as a tangent: one step right, and how much the loss changes |
| `valley_steps.png` | The steps down the valley, and their sizes shrinking |
| `gradient.png` | Each weight's own slope adding up to the gradient, and the step against it |
| `learning_rates.png` | The path across the bowl for 0.03, 0.15 and 0.34 |
| `rate_losses.png` | The loss after each step for the same three learning rates |

## Files

| File | Its one job |
|------|-------------|
| `valley.py` | `Valley`: one weight, its loss and its slope |
| `bowl.py` | `Weights`, `Gradient`, and the `Bowl`: two weights, their loss and gradient |
| `feel.py` | Measuring a slope or a gradient by nudging the weights, no formula needed |
| `descent.py` | `GradientDescent` steps against the slope; `ValleyDescent` and `BowlDescent` record the trips |
| `report.py` | The lines the program prints |
| `valley_figures.py` | The one-weight pictures, in the blog's colours |
| `bowl_figures.py` | The two-weight pictures, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
