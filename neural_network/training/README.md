# Training

`Trainer(backpropagation, gradient_descent)` puts the pieces together.

| Method | What it does |
|--------|--------------|
| `step(network, inputs, truths)` | Finds every slope with backpropagation, then moves every weight and bias one gradient descent step |
| `loss(network, examples)` | The average loss over the examples |
| `train(network, examples, rounds)` | Takes one step for each example, round after round. Returns the loss before training and after each round. |

```python
from neural_network.backpropagation import Backpropagation
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer

trainer = Trainer(Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))
losses = trainer.train(mango_network, [([1.0, 0.5], [1.0])], rounds=40)
# the loss falls from 0.17 to 0.0014, and the answer rises from 0.59 to 0.96
```

`tests/test_training.py` also trains a small network to learn XOR from scratch.

Uses: `backpropagation`, `gradient_descent`, `forward_pass`.

Article: [Backpropagation part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/).

## See it work

```bash
uv run pytest tests/test_training.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_training.py::test_40_rounds_end_96_percent_sure PASSED
      after round 0   loss 0.1699
      after round 1   loss 0.0747
      after round 2   loss 0.0427
      after round 10   loss 0.0074
      after round 40   loss 0.0014
      the answer is now [0.9624]
```
