# artificial-intelligent

Companion code for the AI articles on [byteblueprints.com](https://byteblueprints.com).
Each folder is one project, built in order, with the article's own numbers.
See [CONVENTIONS.md](CONVENTIONS.md) for how every project is built.

```bash
uv sync
uv run python -m ai_01_perceptron   # run one project
uv run pytest                       # check every project against its article
```

| Project | Article |
|---------|---------|
| [ai_01_perceptron](ai_01_perceptron) | [The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/) |
| [ai_02_activations](ai_02_activations) | [Activation functions](https://byteblueprints.com/artificial-intelligence/ann/05_activations/) |
| [ai_03_layers](ai_03_layers) | [Layers and depth](https://byteblueprints.com/artificial-intelligence/ann/06_layers_depth/) |
| [ai_04_forward_pass](ai_04_forward_pass) | [Forward pass](https://byteblueprints.com/artificial-intelligence/ann/07_forward_pass/) |
| [ai_05_loss](ai_05_loss) | [Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/) |
| [ai_06_gradient_descent](ai_06_gradient_descent) | [Gradient descent](https://byteblueprints.com/artificial-intelligence/ann/09_gradient_descent/) |
| [ai_07_backpropagation](ai_07_backpropagation) | Backpropagation, [part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/) and [part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/) |
