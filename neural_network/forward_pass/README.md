# Forward pass

`ForwardPass(network, inputs)` sends the inputs through the network, one
layer at a time, the same way `Network.answer` does. The difference is that it
keeps every step, because training needs them later.

| Class | What it holds |
|-------|---------------|
| `ForwardPass` | `steps`: one `LayerStep` per layer, and `answer`: the last layer's outputs |
| `LayerStep` | What the layer got (`inputs`), each neuron's total before the bend (`totals`), and what came out (`outputs`) |

```python
from neural_network.forward_pass import ForwardPass

forward = ForwardPass(mango_network, [1.0, 0.5])
hidden, output = forward.steps
hidden.totals    # [0.7, 0.55, -0.05]
hidden.outputs   # [0.7, 0.55, 0]: ReLU turned the -0.05 into 0
forward.answer   # [0.588...]
```

`mango_network` is the network in `tests/conftest.py`.

Uses: `network`. Used by: `backpropagation`, `training`.

Tests: `tests/test_forward_pass.py`. Article: [Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/).

## See it work

```bash
uv run pytest tests/test_forward_pass.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_forward_pass.py::test_every_layer_step_is_kept PASSED
      hidden layer got [1.0, 0.5]  totals [0.7, 0.55, -0.05]  outputs [0.7, 0.55, 0.0]
      output layer got [0.7, 0.55, 0.0]  totals [0.355]  outputs [0.5878]
```
