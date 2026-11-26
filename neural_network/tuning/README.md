# Tuning

A network's settings are not handed to you: how many neurons, which learning
rate, how many epochs. Tuning is how you find them. You train, you look at the
loss curves, you change one thing, and you train again. This idea holds the
pieces for that loop: reading the two curves of a run, a quick test for the
learning rate, a rate that changes while the network trains, and two ways to
search several settings together.

It builds on `digits`. The `ScheduledTrainer` is a `BatchTrainer`, and the
range test trains with it. `CurveReading`, `Dial` and the two searches are
plain Python and need no package.

| Class | What it is |
|-------|------------|
| `CurveReading(losses, good_enough)` | Reads a run's training and validation loss. `rate_is_too_high()`, `rate_is_too_low()`, `underfits()` and `overfits()` are asked in that order by `says()` |
| `Schedule` | The base class: `rate(step)` gives the learning rate of each step |
| `Steady(learning_rate)` | One rate for the whole run |
| `WarmUpThenDecay(peak, warm_up_steps, total_steps)` | Climbs in a straight line to the peak, then glides down to 0 |
| `GrowingRate(lowest, highest, steps)` | Multiplied by the same number at every step, for the range test |
| `ScheduledTrainer(network, backpropagation, schedule)` | A `BatchTrainer` that asks the schedule for the rate before every step |
| `RateRangeTest(lowest, highest, steps)` | `run(...)` trains for a few hundred steps while the rate grows. `rates`, `losses`, `smoothed()` and `suggested_rate()` |
| `Dial(name, lowest, highest)` | One setting to tune. `evenly(count)` and `anywhere(random)` both count in times, not plus |
| `Search` | The base class: `trials()` gives one value for every dial, as a dictionary, for each trial |
| `GridSearch(dials, values_per_dial)` | Every way of putting a few values of each dial together |
| `RandomSearch(dials, count, random)` | A fresh random value for every dial in every trial |

## What the curves say

`CurveReading` asks four questions, and the first yes is the thing to fix.

| Question | The sign in the curves | The fix |
|----------|------------------------|---------|
| Is the rate too high? | The training loss is not a number, or it went up in 3 of every 10 epochs and has not halved | Lower the rate |
| Is the rate too low? | The loss is still above `good_enough`, and still fell by a fifth in the second half of the run | Raise the rate |
| Does it underfit? | The loss is still above `good_enough`, and it has gone flat | A bigger network |
| Does it overfit? | The validation loss ends more than a tenth above its best | More data, weight decay, dropout, or stopping early |

Five real runs of the digit reader, each for 10 epochs unless said:

| Run | Training loss | Validation loss | `says()` |
|-----|---------------|-----------------|----------|
| 64 neurons, rate 0.1 | 0.039 | 0.086 | healthy |
| rate 2 | 2.32 | 2.32 | learning rate too high |
| rate 0.001 | 0.435 | 0.398 | learning rate too low |
| 2 neurons | 0.987 | 0.966 | underfitting |
| 500 pictures, 60 epochs | 0.014 | 0.544, after a best of 0.489 | overfitting |

```python
from neural_network.tuning import CurveReading

losses = trainer.train_and_validate(training.inputs, training.truths, validation.inputs, validation.truths,
                                    epochs=10, batch_size=32, shuffle=shuffle)
CurveReading(losses, good_enough=0.1).says()          # "healthy"
```

## The learning rate, in one short run

```python
from neural_network.tuning import RateRangeTest

test = RateRangeTest(lowest=0.0001, highest=10.0, steps=300)
test.run(DigitReader(64, Random(1)), backpropagation, training.inputs, training.truths, batch_size=32, shuffle=shuffle)
test.suggested_rate()                                 # 0.046: a tenth of 0.46, where the loss was lowest
```

On the digit reader the loss does not move below a rate of 0.01, falls from
there to about 0.5, and blows up past 1.

A schedule changes the rate while the network trains:

```python
from neural_network.tuning import ScheduledTrainer, WarmUpThenDecay

schedule = WarmUpThenDecay(peak=0.3, warm_up_steps=500, total_steps=10 * 1563)
trainer = ScheduledTrainer(reader, backpropagation, schedule)
```

With 64 neurons and 10 epochs, a steady rate of 0.3 leaves the validation
loss wobbling between 0.09 and 0.11 and reads 9,759 of the 10,000 validation
pictures right. Warm up and decay to the same peak settles at 0.080 and reads
9,775.

## Searching two dials

```python
from neural_network.tuning import Dial, GridSearch, RandomSearch

dials = [Dial("learning_rate", 0.001, 1.0), Dial("hidden_neurons", 16, 256)]
GridSearch(dials, values_per_dial=3).trials()         # 9 trials, 3 learning rates
RandomSearch(dials, count=9, random=Random(1)).trials()   # 9 trials, 9 learning rates
```

With a short run for each trial (2 epochs on 10,000 pictures), the grid's best
trial reads 9,205 validation pictures right. Random search with the same 9
trials reads 9,359, and over 12 different seeds it beat the grid 9 times.

The tuned reader has 256 neurons, a steady rate of 0.3 and 12 epochs. It
reads 9,836 validation pictures right, up from 9,743, and 9,826 of the 10,000
test pictures, up from 9,774.

`tests/test_tuning.py` checks each reading on curves written by hand, each
schedule step by step, both searches, and the range test on the real pictures.

`tools/export_tuning_runs.py` makes every run above and writes the numbers
the article draws.

Uses: `digits`, `arrays`, `batches`, `training`.

Article: [Tuning and reading loss curves](https://byteblueprints.com/artificial-intelligence/ann/22_tuning/).

## See it work

```bash
uv run pytest tests/test_tuning.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_tuning.py::test_the_curves_say_what_to_fix_first[overfitting] PASSED
      training loss ends at 0.01   went up in 0 of 10 epochs
      validation loss was best at epoch 5   best 0.49   ends at 0.58
      says: overfitting

tests/test_tuning.py::test_warm_up_climbs_to_the_peak_then_glides_down PASSED
      warm up [0.05, 0.1, 0.15, 0.2]
      decay   [0.2, 0.1924, 0.1707, 0.1383, 0.1, 0.0617, 0.0293, 0.0076]

tests/test_tuning.py::test_a_grid_tries_few_values_of_each_dial_and_random_tries_many PASSED
      grid   9 trials, 3 learning rates: [0.001, 0.0316, 1.0]
      random 9 trials, 9 learning rates: [0.0019, 0.0025, 0.0049, 0.0217, 0.0306, 0.0901, 0.1936, 0.1956, 0.3216]
```
