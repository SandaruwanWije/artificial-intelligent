# Activations

An activation is the bend at the end of a neuron. The neuron adds up its
inputs into a total, called z. The activation turns z into the neuron's
output. Without a bend, any number of layers could only ever draw a
straight line.

Every activation also knows its **slope**: how steep it is at z. Training
needs the slope to know which way to move each weight.

| Class | Output | Used for |
|-------|--------|----------|
| `Activation` | The shape every activation follows: `value(z)` and `slope(z)` | |
| `Step` | 1 above 0, else 0 | The perceptron |
| `Sigmoid` | A smooth step from 0 to 1 | Answers that are a chance |
| `Tanh` | A smooth step from -1 to 1 | |
| `ReLU` | 0 below 0, z above 0 | Hidden layers |
| `Gelu` | A smooth ReLU with a small dip | Large language models |
| `NoBend` | z, unchanged | Outputs that can be any number |

Each one has a ready made copy: `STEP`, `SIGMOID`, `TANH`, `RELU`, `GELU`, `NO_BEND`.

```python
from neural_network.activations import SIGMOID

SIGMOID(0)        # 0.5
SIGMOID.slope(0)  # 0.25
```

Tests: `tests/test_activations.py`. Article: [Activation functions](https://byteblueprints.com/artificial-intelligence/ann/05_activations/).

## See it work

```bash
uv run pytest tests/test_activations.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_activations.py::test_sigmoid_is_half_at_zero_and_steepest_there PASSED
      total -3.0 -> value 0.0474   slope 0.0452
      total 0.0 -> value 0.5   slope 0.25
      total 3.0 -> value 0.9526   slope 0.0452
```
