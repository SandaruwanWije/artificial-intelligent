# Network

| Class | What it is |
|-------|------------|
| `Layer` | Neurons side by side. Every neuron gets the same inputs, and the layer gives one output per neuron. Training asks a layer four things, and a kind of layer can answer them in its own way: `inputs_of(n, inputs)`, what neuron `n` reads; `blame_on_inputs(inputs, blame_at_totals)`, how blame goes back to the inputs; and `see_inputs` and `see_blame`, which show it a whole batch. A plain layer has no use for the last two. |
| `Network` | Layers in a row. Each layer's outputs are the next layer's inputs. `answer(inputs)` gives the last layer's outputs. |

A hidden layer is what lets a network do what one neuron cannot. Two ReLU
neurons and one output neuron are enough for XOR:

```python
from neural_network.activations import NO_BEND, RELU
from neural_network.network import Layer, Network
from neural_network.neuron import Neuron

hidden = Layer([Neuron([1, 1], 0, RELU), Neuron([1, 1], -1, RELU)])
output = Layer([Neuron([1, -2], 0, NO_BEND)])
xor = Network([hidden, output])
xor.answer([1, 0])   # [1]
xor.answer([1, 1])   # [0]
```

Uses: `neuron`. Used by: `forward_pass`, `backpropagation`, `training`, `normalization`.

Tests: `tests/test_network.py`. Article: [Layers and depth](https://byteblueprints.com/artificial-intelligence/ann/06_layers_depth/).

## See it work

```bash
uv run pytest tests/test_network.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_network.py::test_a_hidden_layer_solves_xor PASSED
      [0, 0] -> hidden layer [0.0, 0.0] -> output layer [0.0]   want 0
      [0, 1] -> hidden layer [1.0, 0.0] -> output layer [1.0]   want 1
      [1, 0] -> hidden layer [1.0, 0.0] -> output layer [1.0]   want 1
      [1, 1] -> hidden layer [2.0, 1.0] -> output layer [0.0]   want 0
```
