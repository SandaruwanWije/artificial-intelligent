# Loss

A loss turns "how wrong is the answer" into one number. 0 means perfect.

| Class | Cost of one answer | Used for |
|-------|--------------------|----------|
| `Loss` | The shape every loss follows: `cost`, `slope` and `total` | |
| `SquaredError` | The miss times itself. Twice the miss costs four times as much. | Answers that are any number |
| `CrossEntropy` | The surprise at the truth: minus the log of the chance the answer gave it | Answers that are a chance between 0 and 1 |

`slope(answer, truth)` says how the cost moves when the answer moves a little.
Backpropagation starts from it.

```python
from neural_network.loss import CrossEntropy, SquaredError

SquaredError().cost(0.59, 1)   # 0.1681
CrossEntropy().cost(0.99, 1)   # 0.01: sure and right, hardly surprised
CrossEntropy().cost(0.01, 1)   # 4.6: sure and wrong, shocked
```

Tests: `tests/test_loss.py`. Article: [Loss functions](https://byteblueprints.com/artificial-intelligence/ann/08_loss/).

## See it work

```bash
uv run pytest tests/test_loss.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_loss.py::test_cross_entropy_is_small_when_sure_and_right_and_huge_when_sure_and_wrong PASSED
      answer 0.5 truth 1 -> cost 0.6931
      answer 0.99 truth 1 -> cost 0.0101
      answer 0.01 truth 1 -> cost 4.6052
      answer 0.01 truth 0 -> cost 0.0101
```
