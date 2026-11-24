# AI #05: Loss functions

Companion code for the article
[Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/).
It uses the article's words and numbers, so you can read the two side by side.

A loss function compares the network's answer with the truth and turns the
gap into one number. Zero means perfect, bigger means worse. The article
compares two of them:

```
squared error = (truth - answer) ** 2
cross entropy = log(1 / share the network gave to the right answer)
```

They agree when the network is right. When it is sure and wrong they split:
an answer of 0.01 when the truth is yes costs 0.98 in squared error, which can
never go above 1, and 4.6 in cross entropy, which keeps climbing.

A loss over a model's two numbers is a landscape. For a line through the
article's ten points, every (slope, crossing point) is one spot on a map: a
poor line costs 0.47, the best line, slope 0.86 and crossing point 0.32,
costs 0.11. Cross entropy gives the same kind of map for a yes or no rule,
over its weight and bias.

## Run it

From the repo root:

```bash
uv sync
uv run python -m ai_05_loss
uv run pytest ai_05_loss
```

It prints each chapter of the article and saves five pictures to `images/`:

| Picture | Shows |
|---------|-------|
| `squares.png` | Each miss as a square: twice the miss is four times the cost; a poor line and the best one |
| `two_losses.png` | Both losses against the answer, for a truth of yes and of no |
| `surprise.png` | The surprise for each bet, and the same 0.69 added at each halving |
| `landscape.png` | The valley over the slope, then the loss as a map for lines and for yes or no rules |
| `ground.png` | The yes or no rule's ground in 3D: flat for squared error, a steep wall for cross entropy |

## Files

| File | Its one job |
|------|-------------|
| `examples.py` | The ten points and the eight yes or no examples |
| `models.py` | A `Line` (slope, crossing) and a yes or no `Rule` (weight, bias) |
| `losses.py` | `Loss`, and the two that share its methods: `SquaredError`, `CrossEntropy` |
| `landscape.py` | A loss over a model's two numbers: the loss at a spot, the map, its lowest spot |
| `article.py` | The article's numbers, in one place |
| `report.py` | The lines the program prints |
| `figures.py` | The pictures for squared error and cross entropy |
| `maps.py` | The pictures for the landscape: a valley, two maps and the 3D ground |
| `__main__.py` | The article's chapters, in order |
| `tests/` | The article's numbers, checked |
