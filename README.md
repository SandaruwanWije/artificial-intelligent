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
| [forward_pass](neural_network/forward_pass) | `ForwardPass`, `LayerStep`, `BatchForwardPass` | Inputs through the network, keeping every step. A batch goes one layer at a time | [Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/) |
| [loss](neural_network/loss) | `Loss`, `SquaredError`, `CrossEntropy` | How wrong the answer is, as one number | [Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/) |
| [gradient_descent](neural_network/gradient_descent) | `GradientDescent` | One step downhill, against the slope | [Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/) |
| [backpropagation](neural_network/backpropagation) | `NeuronSlopes`, `FiniteDifference`, `Backpropagation` | The slope of every weight and bias: slowly by nudging, then in one sweep backwards | [Part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/), [part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/) |
| [batches](neural_network/batches) | `Shuffle`, `Batches`, `BatchSlopes` | A new order, small groups of examples, and the average slope over a group | [Learning rate, epochs, batch size](https://byteblueprints.com/artificial-intelligence/ann/11_hyperparameters/) |
| [optimizers](neural_network/optimizers) | `Optimizer`, `Momentum`, `RMSProp`, `Adam`, `NeuronOptimizers` | Rules that turn a weight's slope into its step, each with a memory for one weight | [SGD, Momentum, RMSProp, Adam](https://byteblueprints.com/artificial-intelligence/ann/12_optimizers/) |
| [splits](neural_network/splits) | `Split`, `CrossValidation` | Training, validation and test piles, and folds that take turns | [Train, validation, test splits](https://byteblueprints.com/artificial-intelligence/ann/13_splits/) |
| [weight_decay](neural_network/weight_decay) | `WeightDecay`, `NO_DECAY` | A price for large weights, which pulls every weight toward 0 | [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/) |
| [dropout](neural_network/dropout) | `Dropout`, `NO_DROPOUT` | Random hidden neurons switched off in training | [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/) |
| [augmentation](neural_network/augmentation) | `Augmentation` | Nudged copies of the examples | [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/) |
| [early_stopping](neural_network/early_stopping) | `EarlyStopping`, `NeuronNumbers` | Stop when the validation loss stops improving, and keep the best network | [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/) |
| [initialization](neural_network/initialization) | `Initialization`, `Xavier`, `He` | Random starting weights, sized by the number of inputs | [Weight initialization](https://byteblueprints.com/artificial-intelligence/ann/16_init/) |
| [gradient_clipping](neural_network/gradient_clipping) | `GradientClipping`, `NO_LIMIT` | A limit on how long all the slopes together may be | [Vanishing and exploding gradients](https://byteblueprints.com/artificial-intelligence/ann/17_gradients/) |
| [normalization](neural_network/normalization) | `Normalization`, `LayerNorm`, `BatchNorm` | Layers that give a group of numbers a mean of 0 and a spread of 1, then a learned scale and shift | [Batch norm and layer norm](https://byteblueprints.com/artificial-intelligence/ann/18_normalization/) |
| [training](neural_network/training) | `Trainer`, `Losses` | Slopes, then a step, batch after batch and epoch after epoch | [Part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/), [Learning rate, epochs, batch size](https://byteblueprints.com/artificial-intelligence/ann/11_hyperparameters/), [SGD, Momentum, RMSProp, Adam](https://byteblueprints.com/artificial-intelligence/ann/12_optimizers/), [Train, validation, test splits](https://byteblueprints.com/artificial-intelligence/ann/13_splits/), [Vanishing and exploding gradients](https://byteblueprints.com/artificial-intelligence/ann/17_gradients/) |

See [CONVENTIONS.md](CONVENTIONS.md) for the rules every piece follows.
