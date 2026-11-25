# artificial-intelligent

A small neural network library in plain Python, with no other packages. It is
built one idea at a time: each folder in `neural_network/` is one idea, with
its own README. The pieces fit together into a network that learns.

## Run the tests

You need [uv](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync
uv run pytest
```

The tests in `tests/` use the real pieces together, so they also show how to
use them. Each test prints its steps under its name, so you can watch the
numbers move inside a piece. Run one file to look at one idea:

```bash
uv run pytest tests/test_perceptron.py
```

```text
tests/test_perceptron.py::test_one_wrong_guess_moves_every_number PASSED
      before: weights [-0.5, 1.0] bias -0.25
      inputs [0, 1] total 0.75 guess 1 want 0
      error = want - guess = -1   nudge = 0.5 x error = -0.5
      each weight moves by nudge x its input, the bias by the nudge
      after:  weights [-0.5, 0.5] bias -0.75
```

Add `-q` to see only the result.

## The pieces

| Folder | Classes | What it is | Article |
|--------|---------|------------|---------|
| [neuron](neural_network/neuron) | `Neuron` | Weights, a bias and an activation | [What is a neural network?](https://byteblueprints.com/artificial-intelligence/ann/02_what_is_nn/) |
| [perceptron](neural_network/perceptron) | `Perceptron` | A neuron with the step, and its own learning rule | [The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/) |
| [activations](neural_network/activations) | `Activation`, `Step`, `Sigmoid`, `Tanh`, `ReLU`, `Gelu`, `NoBend` | The bend at the end of a neuron, and its slope | [Activation functions](https://byteblueprints.com/artificial-intelligence/ann/05_activations/) |
| [network](neural_network/network) | `Layer`, `Network` | Neurons side by side, and layers in a row | [Layers and depth](https://byteblueprints.com/artificial-intelligence/ann/06_layers_depth/) |
| [forward_pass](neural_network/forward_pass) | `ForwardPass`, `LayerStep` | Inputs through the network, keeping every step | [Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/) |
| [loss](neural_network/loss) | `Loss`, `SquaredError`, `CrossEntropy` | How wrong the answer is, as one number | [Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/) |
| [gradient_descent](neural_network/gradient_descent) | `GradientDescent` | One step downhill, against the slope | [Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/) |
| [backpropagation](neural_network/backpropagation) | `Backpropagation`, `NeuronSlopes` | The slope of every weight and bias, in one sweep backwards | [Part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/), [part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/) |
| [training](neural_network/training) | `Trainer` | Slopes, then a step, round after round | [Part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/) |

See [CONVENTIONS.md](CONVENTIONS.md) for the rules every piece follows.
