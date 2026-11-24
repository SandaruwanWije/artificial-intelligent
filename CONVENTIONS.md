# How every project is built

Each `ai_NN_*` folder is the companion code for one article on
[byteblueprints.com](https://byteblueprints.com). Folders are numbered in the
order they were built; that need not match the article numbers.
`ai_01_perceptron` is the reference: when in doubt, do what it does.

## Layout

```
ai_NN_topic/
  __init__.py      empty
  __main__.py      the article as a program: one function per chapter, then the pictures
  <model>.py       the idea itself, as small classes (one file per concept)
  report.py        turns results into printed lines; never does maths
  figures.py       turns results into pictures; never does maths
  images/          the pictures it saves (committed, the article uses them)
  tests/test_<topic>.py   the article's numbers, checked
  README.md        what it shows, how to run it, file table, link to the article
common/theme.py    the blog's colours and plot helpers, shared by every project
```

Run a project from the repo root with `uv run python -m ai_NN_topic`, and
every test with `uv run pytest`.

## Code

- **Speak the article's language.** Class, method and variable names use the
  article's own words (`weights`, `bias`, `total`, `guess`, `answer`, `error`,
  `learning_rate`, `pass`), and its numbers: the same starting weights, inputs and
  learning rate, so every printed number can be found in the article.
- **Small classes with one job.** Values that do not change are
  `@dataclass(frozen=True)` (`Weights`, `Corner`). Things that act are classes
  with a few short methods (`Perceptron`, `Trainer`). A result is a dataclass
  (`TrainingRun`), not a tuple of loose values.
- **Model, report and figures stay apart.** The model never prints or draws.
  `report.py` and `figures.py` only read results.
- **Plain Python first.** Use NumPy only where the article itself talks about
  vectors or matrices; then name the shapes in comments.
- **The code explains itself, not comments.** Names say what a thing is or
  does (`learning_rate`, `wrong_corners`, `after_each_pass`), and short
  functions say the steps. No comment that repeats what a line does. A module
  gets at most a one-line docstring; a class or function gets none unless its
  name truly cannot say it. The rare comment left is a short *why*. The
  explaining happens in the article's Project tab, around the code.
- Type hints on every function.
- No file over ~150 lines. Split along meaning, not to hit a number.
- No new dependencies: Python 3.12, NumPy and Matplotlib are enough.

## Pictures

- Draw with `common.theme`: `panels()` for a row of plots, `quiet()` for
  axes, `save()` to write the PNG. Use `COLORS`, never raw colour codes:
  `class_0` (plain ink) for answer 0, `accent` (purple) for answer 1 and for
  what the picture points at, `error` (orange) for wrong answers, losses and
  slopes.
- Every picture answers one question the article asks, and its title says
  what to look at. Prefer a row of small panels to one crowded plot.
- Write titles and labels in the article's words.

## Tests

One test per number the article quotes: a starting value, a step, a final
result. If the article changes, a failing test says which sentence to fix.
Tests are for the repo; the article's Project tab does not show them.

## Writing the Project tab

The blog post holds a `<ProjectVersion repo="…/tree/main/ai_NN_topic">`
block after its `</TextVersion>`. Inside it:

1. One short paragraph: what the code is and that it uses the article's numbers.
2. `## Run it` with the `uv sync` and `uv run python -m ai_NN_topic` commands.
3. `## How the code is laid out`: a file table, and a sentence on why it is split that way.
4. Then one section per article chapter, using the chapter's own heading. Each
   section quotes the important code **word for word** from the repo, says in
   plain words what it does and why it is written that way, and shows the
   output that part prints and the picture it draws, placed where the text
   talks about them.

The tab never mentions tests: no test section, no `tests/` row, no test
snippet, no `pytest`. Tests live in the repo only.

Pictures are copied into the post's `project/` folder and shown with
`![alt text that describes the picture](./project/name.png)`.

Check the tab before publishing:

```bash
uv run python tools/check_article.py <path to the post's index.mdx> ai_NN_topic
```

It fails if any code block is not in the project's files, or any output
block is not in what the program prints.
