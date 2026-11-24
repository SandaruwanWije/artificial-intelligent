# AI #04: The forward pass

Companion code for the article
[Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/).
It uses the article's words and numbers, so you can read the two side by side.

A forward pass is what a network does every time you use it. Numbers go in on
one side, each layer turns them into new numbers, and an answer comes out the
other side. Every neuron makes the same three moves:

```
z = each input times the weight on its wire, added up, plus the bias
value = activation(z)          (ReLU in the hidden layer, sigmoid at the output)
```

The network has two inputs, a hidden layer of three ReLU neurons and one
sigmoid output, with 13 hand-picked weights and biases. Feed it the mango
x1 = 1 (how yellow) and x2 = 0.5 (how soft): the hidden layer gives 0.7, 0.55
and 0, and the output gives 0.59.

The code does that pass twice: neuron by neuron in plain Python, and a whole
layer at a time as one matrix multiplication with NumPy,
`h = ReLU(W1 . x + b1)`. A test checks that both give the same numbers.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_04_forward_pass
uv run pytest ai_04_forward_pass
```

It prints each chapter of the article and saves five pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `network.png` | The network with every weight, bias and value written on it |
| `sums.png` | Each neuron's sum built up step by step, then its activation |
| `matrix.png` | Both layers as one matrix move: W · x + b, then the activation |
| `work.png` | The 9 multiplications: 9 steps on one CPU core, 2 on a GPU |
| `gap.png` | The answer, 0.59, against the right answer, 1: 0.41 short |

## Files

| File | Its one job |
|------|-------------|
| `neuron.py` | `Neuron` and its three moves; `Working`, one neuron's sum written out; ReLU and sigmoid |
| `network.py` | `Layer` and `Network`; `forward` runs the pass neuron by neuron into a `ForwardPass` |
| `mango.py` | The article's network, its weights, and the mango it looks at |
| `matrix.py` | The same pass with NumPy: one matrix multiplication per layer |
| `work.py` | Counting multiplications, and how many steps a `Chip` needs for them |
| `race.py` | Times a wide layer both ways (the numbers change every run) |
| `answer.py` | `Gap`: how far the answer falls short of the right one |
| `report.py` | The lines the program prints |
| `figures.py`, `diagrams.py` | The charts, and the network and matrix diagrams, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
