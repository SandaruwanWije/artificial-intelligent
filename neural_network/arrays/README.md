# Arrays

The rest of the library works out a network one number at a time, with loops.
That is the clearest way to see what happens, and it is slow. This idea holds
the same network as NumPy arrays: the examples as the rows of one array, and
each layer's weights as the columns of another. A whole layer, for every
example, is then one line.

It builds on the plain pieces. An `ArrayLayer` is made from a `Layer`, and the
`ArrayTrainer` steps with the same `GradientDescent`. The numbers it gives are
the plain library's numbers.

| Class | What it is |
|-------|------------|
| `ExampleArrays(examples)` | The examples as two arrays: `inputs` and `truths`, one row for each example |
| `ArrayLayer(layer, activation)` | A layer as two arrays: `weights` with one column for each neuron, and `biases` |
| `ArrayNetwork(layers)` | Layers in a row. `answer(inputs)` gives every example's answer |
| `ArrayActivation` | The base class: `value(totals)` and `slope(totals)` for a whole array. `ArrayReLU` and `ArraySigmoid` are its two curves, ready made as `ARRAY_RELU` and `ARRAY_SIGMOID` |
| `ArrayForwardPass(network, inputs)` | Every example forward. Keeps an `ArrayStep` (inputs, totals, outputs) for each layer |
| `ArrayLoss` | The base class: `costs`, `slopes` and `average`. `ArrayCrossEntropy` is the loss for answers between 0 and 1 |
| `ArrayBackpropagation(loss)` | `slopes(network, inputs, truths)` gives one `LayerSlopes` per layer. Each slope is the average over the examples |
| `ArrayTrainer(network, backpropagation, gradient_descent)` | `step`, `loss`, and `train(inputs, truths, epochs)`, with every example in each step |

The shapes are the thing to follow. With 36 examples, 2 inputs and 3 hidden
neurons:

| Array | Shape |
|-------|-------|
| `inputs` | (36, 2) |
| hidden `weights` | (2, 3) |
| hidden `totals`, `inputs @ weights + biases` | (36, 3) |
| output `weights` | (3, 1) |
| the answers | (36, 1) |
| hidden weight slopes, `inputs.T @ at_totals` | (2, 3), like the weights they move |

```python
from neural_network.arrays import (ARRAY_RELU, ARRAY_SIGMOID, ArrayBackpropagation, ArrayCrossEntropy,
                                   ArrayLayer, ArrayNetwork, ArrayTrainer, ExampleArrays)
from neural_network.gradient_descent import GradientDescent

hidden, output = network.layers                      # a plain Network: 2 inputs, 3 ReLU, 1 sigmoid
arrays = ArrayNetwork([ArrayLayer(hidden, ARRAY_RELU), ArrayLayer(output, ARRAY_SIGMOID)])
mangoes = ExampleArrays(examples)                    # inputs (36, 2), truths (36, 1)

trainer = ArrayTrainer(arrays, ArrayBackpropagation(ArrayCrossEntropy()), GradientDescent(learning_rate=0.5))
losses = trainer.train(mangoes.inputs, mangoes.truths, epochs=200)   # 0.6966 down to 0.3323
```

On one laptop, those 200 epochs take about 84 ms with the plain `Trainer` and
about 9 ms with the arrays. With 2,000 examples and 64 hidden neurons, one
epoch takes about 380 ms with loops and 3.3 ms with arrays.

This idea needs a package. NumPy is in the `arrays` dependency
group, which `uv` installs by default. Without NumPy, its tests are skipped.

`tests/test_arrays.py` checks every total, answer and slope against the plain
network, and that 200 epochs end on the plain trainer's numbers.

Uses: `network`, `gradient_descent`.

Article: [ANN from scratch in NumPy](https://byteblueprints.com/artificial-intelligence/ann/19_numpy_ann/).

## See it work

```bash
uv run pytest tests/test_arrays.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_arrays.py::test_one_step_gives_all_three_totals_for_four_mangoes PASSED
      mango [0.61, 0.66]   totals [0.324, 0.577, 0.264]   outputs [0.324, 0.577, 0.264]
      mango [0.06, 0.8]   totals [-0.172, 0.538, 0.664]   outputs [0.0, 0.538, 0.664]
      mango [0.86, 0.47]   totals [0.6, 0.481, 0.019]   outputs [0.6, 0.481, 0.019]
      mango [0.38, 0.41]   totals [0.24, 0.283, 0.277]   outputs [0.24, 0.283, 0.277]

tests/test_arrays.py::test_training_ends_on_the_plain_trainers_numbers PASSED
      loss after 200 epochs: arrays 0.3323   plain 0.3323
      first neuron's weights: arrays [2.3638, 2.2151]   plain [2.3638, 2.2151]
```
