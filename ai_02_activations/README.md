# AI #02: Activation functions

Companion code for the article
[Activation functions](https://byteblueprints.com/artificial-intelligence/ann/05_activations/).
It uses the article's words and numbers, so you can read the two side by side.

The activation function is the bend at the end of every neuron. The neuron
adds up its inputs into one number, z, and the activation turns z into its
output. What matters for learning is the slope: how much the output moves
when z moves a little.

```
step     1 if z > 0, else 0      slope 0 everywhere
sigmoid  1 / (1 + e^-z)          slope 0.25 at most
tanh     tanh(z)                 slope 1 at most
ReLU     max(0, z)               slope 0 or 1
GELU     z * Phi(z)              a smooth ReLU
```

Without a bend, layers only multiply and add, so a stack of them is still
one straight line. Three ReLU neurons with corners at 0, 1 and 2 draw the
article's bump. On the ring of 200 points the same small network gets 57%
right with no bend and 100% with ReLU or tanh.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_02_activations
uv run pytest ai_02_activations
```

It prints each chapter of the article and saves five pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `step_to_sigmoid.png` | The step's slope is 0 everywhere; the sigmoid has one at every z |
| `four_curves.png` | Sigmoid, tanh, ReLU and GELU with their slopes, flat stretches shaded |
| `no_bend.png` | One layer and two layers with no bend: both straight lines |
| `relu_bump.png` | Three ReLU neurons, each with one corner, and the bump they add up to |
| `ring.png` | The same network on the ring with no bend, ReLU and tanh |

## Files

| File | Its one job |
|------|-------------|
| `activations.py` | Every activation as a class with a value and a slope |
| `layers.py` | `Layer`, a layer with no bend, and how two of them become one |
| `bump.py` | The bump drawn by hand: three ReLU neurons and a final neuron |
| `seeded.py` | The blog's seeded random numbers, so the ring and starting weights match |
| `ring.py` | The 200 points of the ring, and how many a network gets right |
| `network.py` | `TinyNetwork`: 4 hidden neurons with any activation, one sigmoid output |
| `trainer.py` | `Trainer` runs the same training for every activation; `TrainingRun` records it |
| `report.py` | The lines the program prints |
| `figures.py`, `figures_bump.py`, `figures_ring.py` | The pictures, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
