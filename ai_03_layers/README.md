# AI #03: Layers and depth

Companion code for the article
[Layers and depth](https://byteblueprints.com/artificial-intelligence/ann/06_layers_depth/).
It uses the article's words and numbers, so you can read the two side by side.

One neuron draws one straight line, and no single line can split XOR's
corners. Put a hidden layer of two ReLU neurons in front of the output and
it works:

```
h1 = max(0, x1 + x2)
h2 = max(0, x1 + x2 - 1)
y  = h1 - 2 * h2
```

The hidden layer gives every corner new numbers, (h1, h2). Both corners that
want 1 land on (1, 0), and one straight line is enough.

On a noisy cloud of 160 XOR points, a network with 1 hidden neuron gets 74%
right, 2 neurons get 74% or 97% depending on the random start, and 3, 4 or 8
get every point. Last, two ReLU neurons make a tent; stack them layer after
layer and the corners double each time: 1, 3, 7, 15. Ten layers of two
neurons make 1,023 corners, which one wide layer needs 1,023 neurons for.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_03_layers
uv run pytest ai_03_layers
```

It prints each chapter of the article and saves five pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `one_line.png` | Five lines one neuron tries, each with a wrong corner, and the two lines that work |
| `new_numbers.png` | The corners before the hidden layer, and at their new numbers (h1, h2) |
| `cloud_two.png` | 1 and 2 hidden neurons on the cloud, and 2 again from another start |
| `cloud_more.png` | 3, 4 and 8 hidden neurons, every point right |
| `tents.png` | The tent after 1 to 4 layers, and the neurons 15 corners take each way |

## Files

| File | Its one job |
|------|-------------|
| `neuron.py` | The activations, and a `Neuron`: weighted sum, bias, answer |
| `network.py` | A `Layer` of neurons, and a `Network` that passes answers layer to layer |
| `xor.py` | XOR's corners, the lines one neuron tries, and the network that solves it |
| `cloud.py` | The noisy XOR cloud, from the same seeded random source as the article |
| `trainer.py` | `Trainer` trains a hidden layer on the cloud; a `Trial` is one trained network |
| `tents.py` | The tent, stacked layer after layer, and its corners counted |
| `report.py` | The lines the program prints |
| `figures.py`, `figures_cloud.py` | The pictures, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
