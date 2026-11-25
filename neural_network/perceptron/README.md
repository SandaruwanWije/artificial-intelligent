# Perceptron

A perceptron is a `Neuron` whose activation is the `Step`. It has weights, a
bias and a hard edge: it answers 1 when its total is above 0, and 0
otherwise. It also has its own way to learn.

| Method | What it does |
|--------|--------------|
| `guess(inputs)` | The answer: 0 or 1 |
| `learn(inputs, answer)` | Works out the error (1, 0 or -1). Each weight moves by the learning rate times the error times its own input, and the bias by the learning rate times the error. A right guess moves nothing. |
| `train(examples)` | Goes round the examples, pass after pass, until a pass has no mistakes. Returns the mistakes in each pass. |

```python
from neural_network.perceptron import Perceptron

AND = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]

perceptron = Perceptron(weights=[-0.5, 1.0], bias=-0.25, learning_rate=0.5)
perceptron.train(AND)   # [2, 2, 1, 0]: learned in four passes
```

It can learn AND and OR, but never XOR: one straight line cannot split XOR's corners.

Uses: `neuron`, `activations`.

Tests: `tests/test_perceptron.py`. Article: [The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/).

## See it work

```bash
uv run pytest tests/test_perceptron.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_perceptron.py::test_one_wrong_guess_moves_every_number PASSED
      before: weights [-0.5, 1.0] bias -0.25
      inputs [0, 1] total 0.75 guess 1 want 0
      error = want - guess = -1   nudge = 0.5 x error = -0.5
      each weight moves by nudge x its input, the bias by the nudge
      after:  weights [-0.5, 0.5] bias -0.75
```
