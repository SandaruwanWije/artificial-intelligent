# Early stopping

While a network trains, its training loss keeps falling. Its validation loss
falls too, then stops, and then often climbs: from there on the network is
learning the noise of its own examples. Early stopping ends the training
around that point and keeps the network as it was at its best.

`EarlyStopping(trainer, patience)` wraps a `Trainer`:

| Method | What it does |
|--------|--------------|
| `train(training, validation, epochs, batch_size, shuffle)` | Trains one epoch at a time and measures both losses after each. It remembers the network from the epoch with the lowest validation loss. It stops when `patience` epochs in a row bring no new best, or after `epochs`. Then it puts the best network back. Returns a `Losses` up to the epoch where it stopped. |
| `numbers_of(network)` | A copy of every weight and bias, one `NeuronNumbers` per neuron |
| `put_back(network, numbers)` | Sets every weight and bias of the network to a kept copy |

It copies weights and biases only. A `BatchNorm` layer's kept means and
spreads are not copied, so they stay as the last epoch left them.

After `train`, `best_epoch` says which epoch was kept. Epoch 0 is the network
before any training.

```python
import random

from neural_network.batches import Shuffle
from neural_network.early_stopping import EarlyStopping

stopping = EarlyStopping(trainer, patience=20)
losses = stopping.train(split.training, split.validation, epochs=300, batch_size=6,
                        shuffle=Shuffle(random.Random(5)))
stopping.best_epoch   # 75: the network now holds the weights it had after epoch 75
```

`tests/test_early_stopping.py` stops a run of 300 epochs after 95, and checks
that the network it keeps is exactly the one that 75 epochs give.

Uses: `training`, `batches`, `network`.

Article: [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/).

## See it work

```bash
uv run pytest tests/test_early_stopping.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_early_stopping.py::test_it_stops_20_epochs_after_the_best_one_and_keeps_the_best PASSED
      after epoch 0   training loss 0.2624   validation loss 0.2446
      after epoch 25   training loss 0.1501   validation loss 0.1419
      after epoch 50   training loss 0.1139   validation loss 0.1193
      after epoch 75   training loss 0.1077   validation loss 0.1166
      after epoch 95   training loss 0.1085   validation loss 0.1334
      best epoch 75   stopped after epoch 95 of 300
      the network that is kept has a validation loss of 0.1166
```
