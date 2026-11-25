# Neuron

A neuron has a weight for each input, one bias, and an activation.

1. `total`: multiply each input by its weight, add them up, and add the bias.
2. `output`: bend that total with the activation.

```python
from neural_network.activations import RELU
from neural_network.neuron import Neuron

neuron = Neuron(weights=[0.8, -0.4], bias=0.1, activation=RELU)
neuron.total([1.0, 0.5])   # 0.7
neuron.output([1.0, 0.5])  # 0.7
```

Uses: `activations`. Used by: `perceptron`, `network`.

Tests: `tests/test_neuron.py`. Article: [What is a neural network?](https://byteblueprints.com/artificial-intelligence/ann/02_what_is_nn/)

## See it work

```bash
uv run pytest tests/test_neuron.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_neuron.py::test_the_total_is_each_input_times_its_weight_plus_the_bias PASSED
      input 1.0 x weight 0.8 = 0.8
      input 0.5 x weight -0.4 = -0.2
      add them up, plus the bias 0.1 -> total 0.7
```
