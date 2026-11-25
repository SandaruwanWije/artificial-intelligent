# Data augmentation

More data is the best cure for a network that memorises. When there is no
more data to get, augmentation makes new examples out of the ones there are.

`Augmentation(copies, nudge, random)` has one method:

| Method | What it gives |
|--------|---------------|
| `augmented(examples)` | The examples as they are, followed by `copies` nudged copies of every one. In a copy, each input is moved by a small random amount with a spread of `nudge`. The truths are kept. |

A nudged copy is like measuring the same mango once more: the reading is
close to the first one, but not equal. The change must be small enough that
the truth still holds.

The list it was given is not changed. It takes a `random.Random(seed)`, so
the same seed gives the same copies every run.

```python
import random

from neural_network.augmentation import Augmentation

more = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(examples)
# 24 examples -> 96: the 24 real ones, then 3 nudged copies of each
```

`tests/test_augmentation.py` turns 24 mangoes into 96 and measures how far
the copies moved.

Uses: nothing. Works with: `training`.

Article: [Regularization](https://byteblueprints.com/artificial-intelligence/ann/15_regularization/).

## See it work

```bash
uv run pytest tests/test_augmentation.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_augmentation.py::test_a_copy_keeps_the_truth_and_moves_each_input_a_little PASSED
      the first six moves: [0.073, 0.0056, -0.178, 0.142, 0.0472, 0.0561]
      the usual size of all 144 moves is 0.1608   the nudge asked for is 0.15

tests/test_augmentation.py::test_training_on_the_copies_takes_four_times_the_steps_in_every_epoch PASSED
      24 real mangoes   4 steps in an epoch   loss on the 24 real mangoes 0.1139   validation loss 0.1193
      96 with the copies   16 steps in an epoch   loss on the 24 real mangoes 0.112   validation loss 0.1298
```
