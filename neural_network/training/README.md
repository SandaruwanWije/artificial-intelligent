# Training

`Trainer(network, backpropagation, optimizer)` puts the pieces together. A
trainer trains one network. When it is built, it gives every weight and bias
a fresh optimizer of its own, so each one keeps its own memory.

| Method | What it does |
|--------|--------------|
| `one_for_each_number(optimizer)` | Makes a fresh optimizer for every weight and bias of the network |
| `step(batch)` | Finds every slope, averaged over the batch, adds the weight decay, clips them, then moves every weight and bias one step with its own optimizer |
| `loss(examples)` | The average loss over the examples, with the network as it is used: `Network.answer` gives each answer |
| `epoch(examples, batch_size, shuffle)` | One trip through the examples. It shuffles them, cuts them into batches, and takes one step for each batch. |
| `train(examples, epochs, batch_size, shuffle)` | Runs that many epochs. Returns the loss before training and after each epoch. |
| `train_and_validate(training, validation, epochs, batch_size, shuffle)` | The same, but it also measures the loss on a validation set after each epoch. It never steps on that set. Returns a `Losses`, with the lists `training` and `validation`. |

An example is a pair: its inputs and its truths. A batch is a list of
examples. A batch size of 1 takes one step for every example.

```python
import random

from neural_network.backpropagation import Backpropagation
from neural_network.batches import Shuffle
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.training import Trainer

trainer = Trainer(mango_network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))
losses = trainer.train([([1.0, 0.5], [1.0])], epochs=40, batch_size=1, shuffle=Shuffle(random.Random(5)))
# the loss falls from 0.17 to 0.0014, and the answer rises from 0.59 to 0.96
```

Any optimizer works in place of `GradientDescent`: `Momentum`, `RMSProp` or
`Adam` from `neural_network.optimizers`.

A fourth argument is optional: a `GradientClipping`. With it, `step` shrinks
the slopes to its limit before it moves anything. Without it the trainer uses
`NO_LIMIT`, which never changes a slope.

A fifth is optional too: a `WeightDecay`. With it, `step` adds a pull toward
0 to every weight's slope. Without it the trainer uses `NO_DECAY`, which adds
nothing. Dropout is not set here: it is given to the `Backpropagation`.

`tests/test_training.py` also trains a small network to learn XOR from
scratch. Then it trains one number on a crate of 24 mangoes: with three
learning rates, with 24, 1 and 6 mangoes per step, and for 10 epochs. Last,
it trains the mango network with each of the four optimizers.

`tests/test_splits.py` uses `train_and_validate` on the piles of a `Split`.

Uses: `backpropagation`, `batches`, `optimizers`, `gradient_clipping`, `weight_decay`, `forward_pass`. Used by: `early_stopping`.

Articles: [Backpropagation part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/), [Learning rate, epochs, batch size](https://byteblueprints.com/artificial-intelligence/ann/11_hyperparameters/), [SGD, Momentum, RMSProp, Adam](https://byteblueprints.com/artificial-intelligence/ann/12_optimizers/), [Train, validation, test splits](https://byteblueprints.com/artificial-intelligence/ann/13_splits/), [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/) and [Vanishing and exploding gradients](https://byteblueprints.com/artificial-intelligence/ann/17_gradients/).

## See it work

```bash
uv run pytest tests/test_training.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_training.py::test_40_epochs_on_one_mango_end_96_percent_sure PASSED
      after epoch 0   loss 0.1699
      after epoch 1   loss 0.0747
      after epoch 2   loss 0.0427
      after epoch 10   loss 0.0074
      after epoch 40   loss 0.0014
      the answer is now [0.9624]

tests/test_training.py::test_3_epochs_with_24_then_1_then_6_mangoes_per_step PASSED
      24 per step  3 steps   loss after each epoch [11.2402, 7.5538, 5.1945, 3.6846]
      1 per step  72 steps   loss after each epoch [11.2402, 1.0989, 1.0807, 1.001]
      6 per step  12 steps   loss after each epoch [11.2402, 2.6045, 1.2986, 1.0507]

tests/test_training.py::test_each_optimizer_trains_the_mango_network PASSED
      GradientDescent  loss after 0, 10, 20, 40 epochs: [0.1699, 0.0074, 0.0032, 0.0014]   answer [0.9624]
      Momentum  loss after 0, 10, 20, 40 epochs: [0.1699, 0.0093, 0.0006, 0.0001]   answer [0.9888]
      RMSProp  loss after 0, 10, 20, 40 epochs: [0.1699, 0.0423, 0.0194, 0.006]   answer [0.9228]
      Adam  loss after 0, 10, 20, 40 epochs: [0.1699, 0.0685, 0.0273, 0.008]   answer [0.9105]
```
