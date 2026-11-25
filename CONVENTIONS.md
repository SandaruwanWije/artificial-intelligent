# How the library is built

The repo is one small neural network library. Each new article adds the piece
it teaches. Nothing else goes in: no pictures, no printed reports, no programs
to run. The tests are how you see a piece work.

## Layout

```
neural_network/
  <idea>/
    __init__.py     imports the idea's classes, so users write
                    from neural_network.<idea> import <Class>
    <idea>.py       the classes
    README.md       what the idea is, its classes, a short example, its test file
tests/
  conftest.py       shared test networks (the mango network)
  test_<idea>.py    one file per idea, using the real pieces together
tools/
  check_article.py  checks a blog post's Project tab against the code
```

## Classes

- **Every idea is a class named after it**: `Neuron`, `Perceptron`,
  `Activation`, `Network`, `ForwardPass`, `Loss`, `GradientDescent`,
  `Backpropagation`, `Trainer`. Its methods are what the idea does
  (`total`, `output`, `guess`, `learn`, `slope`, `step`).
- **Build on the pieces that are already there.** A `Perceptron` is a
  `Neuron` with the `Step`. A `Network` is made of `Layer`s of `Neuron`s.
  `Trainer` uses `Backpropagation` and `GradientDescent`. Never write a second
  version of a piece.
- **Families share a base class.** `Activation` and `Loss` are abstract. A new
  curve or loss is a new subclass, never an `if` on a name.
- **Only what the idea needs.** No helpers for printing or drawing, no extra
  options. Keep each class as small as it can be and still be right.
- **No loose functions** in the library. Ready made instances are fine
  (`RELU`, `SIGMOID`).
- Plain Python lists and `math`. No other packages.
- **Plain steps, not clever code.** The library is for seeing what happens,
  not for speed. Use `for` loops with an index, `if`, `append`, and `+ - * /`.
  Give each in between number a name (`total`, `miss`, `at_total`). No list
  comprehensions, `sum`, `zip`, `enumerate`, `max`, `**` or one line `if`
  expressions. `math.exp`, `math.log` and `math.tanh` are fine. A short
  comment may say what a step means.
- Easy names a beginner can guess. One line docstring per class. Type hints.
  Files under about 100 lines.

## The code stands on its own

Anyone should be able to use the library without reading the articles. The
code and the tests never mention an article, the blog or a chapter. Only
README files link to the articles.

## Tests

Each idea has `tests/test_<idea>.py`. A test builds the real pieces and checks
a result someone could work out by hand. Slopes are also checked against a
tiny nudge. `uv run pytest` must pass before anything is committed.

Tests also show how the piece works inside. A test takes the `show` fixture
from `tests/conftest.py` and calls it at each step, for example
`show("after:  weights", perceptron.weights, "bias", perceptron.bias)`.
Each line is printed under the test's name when the tests run (floats are
rounded to 4 places). Show the numbers someone would follow by hand: inputs,
totals, errors, weights before and after. Check the in between numbers with
`assert` too, not only the last one. `uv run pytest -q` hides the steps.

## Writing the Project tab

A blog post holds a `<ProjectVersion>` block after its `</TextVersion>`. It
explains the library code of the piece the post adds, simply and briefly.
Tests and how to run them live only in the README files, never in the tab:
no test code, no test output, no run commands.

1. One or two sentences: which piece this is and what it builds on.
2. `## Where it is`: a short table of the classes and their files.
3. `## Walk through the code`: one to five short steps. Each is a bold lead
   in sentence or two in plain words, then code quoted **word for word** from
   `neural_network/`. Always show the `class` line a method belongs to: when
   the method is further down, write the class line, then a line with only
   `...`, then the method. With only one step, leave out its number.

Plain language: short sentences, everyday words, no idioms, no dashes used
as punctuation. Then check the tab. Every code block must be found in the
library:

```bash
uv run python tools/check_article.py <path to the post's index.mdx>
```
