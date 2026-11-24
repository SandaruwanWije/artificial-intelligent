# AI #01: The perceptron

Companion code for the article
[The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/).
It uses the article's words and numbers, so you can read the two side by side.

A perceptron is one neuron with a hard edge. Each signal is multiplied by its
weight, the results are added up, and the bias is added on top. That total is z:

```
z = w1 * x1 + w2 * x2 + bias
answer = 1 if z > 0 else 0
```

When it gets an example wrong, it nudges each number toward the right answer:

```
error = answer - guess          (1, 0 or -1)
weight <- weight + rate * error * its signal
bias   <- bias + rate * error
```

Every gate starts from the article's poor line (w1 -0.5, w2 1, bias -0.25)
and learns with a rate of 0.5. AND and OR are learned in four passes. XOR
never is: no single straight line can split its corners.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_01_perceptron
uv run pytest ai_01_perceptron
```

It prints each chapter of the article and saves three pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `and_passes.png` | AND's line at the start and after every pass |
| `mistakes.png` | Mistakes per pass; only AND and OR reach zero |
| `boundaries.png` | The line each gate ends on, wrong corners ringed in orange |

## Files

| File | Its one job |
|------|-------------|
| `gates.py` | AND, OR and XOR as data: four corners each |
| `perceptron.py` | `Weights`, and the `Perceptron`: z, the hard edge and the learning rule |
| `trainer.py` | `Trainer` makes passes until one is clean; `TrainingRun` records them |
| `report.py` | The lines the program prints |
| `figures.py` | The pictures, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
