# Splits

A network can score 100% on the examples it trained on just by remembering
them. To grade it honestly, some examples must be hidden from it before
training starts. Two classes do the hiding.

## `Split`: three piles

`Split(examples, validation_size, test_size, shuffle)` shuffles the examples,
then cuts them into three piles. The crate it was given keeps its order.

| Pile | What it is for |
|------|----------------|
| `training` | What the network learns from. It gets every example the other two do not take. |
| `validation` | What the settings are chosen by. The network never trains on it. |
| `test` | Kept away until every choice is made, then used once. |

It shuffles first because data is often stored in order. A crate packed with
the ripe mangoes first would leave no ripe mango in the last two piles.

## `CrossValidation`: every example validates once

`CrossValidation(examples, folds)` cuts the examples into folds of equal
size, or 1 apart when they cannot be equal. It does not shuffle, so give it
examples that a `Split` has already shuffled.

| Method | What it gives |
|--------|---------------|
| `validation(run)` | The fold whose turn it is in this run |
| `training(run)` | Every other fold, put together |

There is one run for each fold. Each run trains a new network from the start.

```python
import random

from neural_network.batches import Shuffle
from neural_network.splits import CrossValidation, Split

split = Split(crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
losses = trainer.train_and_validate(split.training, split.validation, epochs=200, batch_size=6,
                                    shuffle=Shuffle(random.Random(5)))
# losses.training and losses.validation: one loss before training and one after each epoch

cross = CrossValidation(split.training + split.validation, folds=5)   # the test pile stays locked
for run in range(5):
    training, validation = cross.training(run), cross.validation(run)
```

`train_and_validate` is in `training`. It steps on the training set only and
measures the loss on both sets after each epoch.

`tests/test_splits.py` splits a crate of 36 mangoes into 24, 6 and 6, trains
on the first pile while it watches the second, and then runs all five folds.

Uses: `batches`. Works with: `training`.

Article: [Train, validation, test splits](https://byteblueprints.com/artificial-intelligence/ann/13_splits/).

## See it work

```bash
uv run pytest tests/test_splits.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_splits.py::test_a_crate_packed_ripe_first_shows_why_the_shuffle_comes_first PASSED
      cut as packed: ripe in training, validation, test = (18, 0, 0)
      shuffled, then cut:                                (13, 2, 3)

tests/test_splits.py::test_training_keeps_a_loss_for_both_sets_after_each_epoch PASSED
      after epoch 0   training loss 0.2624   validation loss 0.2446
      after epoch 1   training loss 0.2449   validation loss 0.2269
      after epoch 10   training loss 0.1954   validation loss 0.1793
      after epoch 50   training loss 0.1139   validation loss 0.1193
      after epoch 100   training loss 0.1042   validation loss 0.1289
      after epoch 200   training loss 0.0994   validation loss 0.1224
      right: training 20 of 24   validation 5 of 6
      the test pile, opened once at the end: loss 0.165   right 4 of 6

tests/test_splits.py::test_five_runs_grade_the_network_on_30_mangoes_instead_of_6 PASSED
      run 0   validation loss 0.2961   right 3 of 6
      run 1   validation loss 0.1724   right 4 of 6
      run 2   validation loss 0.0784   right 5 of 6
      run 3   validation loss 0.0124   right 6 of 6
      run 4   validation loss 0.1224   right 5 of 6
      all five runs together: 23 of 30
```
