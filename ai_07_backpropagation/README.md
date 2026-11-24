# AI #07: Backpropagation

Companion code for the article Backpropagation, in two parts:
[part 1](https://byteblueprints.com/artificial-intelligence/ann/10_backprop/) and
[part 2](https://byteblueprints.com/artificial-intelligence/ann/10_backprop_2/).
It uses the article's words and numbers, so you can read the two side by side.

The network is article 7's: two inputs, three ReLU neurons and a sigmoid
answer. A mango goes in (x1 = 1, x2 = 0.5), the answer comes out 0.59, and
against the right answer of 1 the loss is 0.17. The network can change 13
numbers, w1 to w9 and b1 to b4, and learning needs a slope for each one.

Part 1 measures one slope with a nudge of 0.01:

```
slope = (loss after the nudge - loss before) / 0.01  =  -0.0024 / 0.01  =  -0.24
```

Part 2 works every slope out instead. First link by link, multiplying one
small ratio for each link on a number's road to the loss:

```
slope of w1 = x1 x ReLU's ratio x w7 x y(1 - y) x 2(y - 1)
            = 1  x 1            x 1.2 x 0.24    x -0.82     = -0.24
```

Then all 13 in one sweep back from the loss, each part's number worked out
once and used by everything behind it. One step with a learning rate of 0.5
takes the loss from 0.17 to 0.075, and 40 rounds take it to 0.0014.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_07_backpropagation
uv run pytest ai_07_backpropagation
```

It prints each chapter of both parts and saves six pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `one_at_a_time.png` | The network, each number coloured by which way it pushes the loss |
| `nudge.png` | The loss as w1 alone moves, and the nudge of 0.01 up close |
| `formulas.png` | All 13 formulas as rows of link ratios, ending in the same two links |
| `sweep.png` | The sweep back: a number at every part and a slope on every wire |
| `agreement.png` | The nudge test and the backward sweep, side by side for all 13 |
| `training.png` | The loss falling and the answer climbing over 40 rounds |

## Files

| File | Its one job |
|------|-------------|
| `network.py` | The 13 `Parameters`, the mango `Example`, and the `Network` with its forward pass |
| `nudge.py` | `NudgeTest`: measure a slope by turning one number up a little |
| `backprop.py` | `Backprop`: each slope link by link, then all 13 in one `Sweep` |
| `trainer.py` | `Trainer` steps against the slopes, round after round; `TrainingRun` records them |
| `report.py`, `report_part2.py` | The lines the program prints, for part 1 and part 2 |
| `drawing.py` | The network as a picture: circles, wires and their labels |
| `network_figures.py`, `figures.py` | The pictures, in the blog's colours |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
