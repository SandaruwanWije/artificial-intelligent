# AI #01: The perceptron

Companion code for the article
[The perceptron](https://byteblueprints.com/artificial-intelligence/ann/04_perceptron/).
The code uses the article's words and numbers, so you can read the two side by side.

A perceptron is one neuron with a hard edge. Each signal is multiplied by its
weight, the results are added up, and the bias is added on top. That total is z:

```
z = w1 * x1 + w2 * x2 + bias
answer = 1 if z > 0 else 0
```

When it gets an example wrong, it nudges each number toward the right answer:

```
error = answer - guess          (1, 0 or -1)
weight <- weight + rate * error * its signal
bias   <- bias + rate * error
```

## Run it

From the repo root:

```bash
uv sync
uv run python ai_01_perceptron/main.py
```

`main.py` has one function per chapter of the article:

| Chapter | What it prints |
|---------|----------------|
| One neuron with a hard edge | z at each corner, with weights set by hand for AND and OR |
| Learning AND | every fix it makes, pass by pass, from the article's poor start |
| Pass after pass: OR | the mistakes in each pass |
| The limit: XOR | that it is still wrong after 8 passes |

It also saves two pictures to `images/`:

| File | Shows |
|------|-------|
| `boundaries.png` | The line each gate ends on, with wrong corners ringed in orange |
| `mistakes.png` | Mistakes per pass; only AND and OR reach zero |

## Files

| File | Purpose |
|------|---------|
| `perceptron.py` | The `Perceptron` class, in plain Python |
| `main.py` | The article's chapters, one function each |
| `plots.py` | The two pictures, in the blog's colours |
