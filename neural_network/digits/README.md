# Digits

The first real problem for the library: reading handwritten digits. The
pictures are MNIST, 70,000 small grey pictures of the digits 0 to 9, each with
the digit it shows. A picture is 28 by 28 pixels, so the network gets 784
inputs. A digit is one of ten things, so it gives ten scores, and softmax turns
the scores into ten chances.

It builds on `arrays`. The `DigitReader` is an `ArrayNetwork`, the loss is an
`ArrayLoss`, and the `BatchTrainer` is an `ArrayTrainer`. The same
`ArrayBackpropagation` finds every slope. Only three things are new: the
pictures, softmax, and steps on small batches.

| Class | What it is |
|-------|------------|
| `MnistFile(path)` | The MNIST file. `download()` fetches it once. `training()`, `validation()` and `test()` give the three piles: 50,000, 10,000 and 10,000 pictures |
| `DigitPictures(pictures, labels)` | Pictures as arrays: `inputs` (one row of 784 numbers from 0 to 1 for each picture), `truths` (one row of ten, a 1 at the true digit) and `labels` |
| `Softmax` | `chances(scores)` turns each row of scores into chances that add up to 1 |
| `SoftmaxCrossEntropy` | The loss for scores: softmax, then the surprise at the chance of the true class. Its slope for every score is the chance minus the truth |
| `ArrayNoBend` | An activation that changes nothing, for a last layer that gives raw scores. Ready made as `ARRAY_NO_BEND` |
| `DigitReader(hidden_neurons, random)` | The network: 784 inputs, a ReLU layer, ten scores. `chances(inputs)`, `digits(inputs)` and `right(pictures)` |
| `BatchTrainer(network, backpropagation, gradient_descent)` | An `ArrayTrainer` with `epoch(inputs, truths, batch_size, shuffle)`: a new order, then one step for every batch. `train_and_validate` keeps both losses after every epoch |
| `Confusion(labels, read)` | `counts[true][read]` for every pair, `right()`, and `worst_mix_up()` |

The shapes, with 64 hidden neurons and a batch of 32 pictures:

| Array | Shape |
|-------|-------|
| a batch of `inputs` | (32, 784) |
| hidden `weights` | (784, 64) |
| score `weights` | (64, 10) |
| the scores, and the chances | (32, 10) |
| numbers to learn | 784 × 64 + 64 + 64 × 10 + 10 = 50,890 |

## Get the pictures

The file is not part of the repo. Fetch it once, about 11 MB, into `data/`:

```bash
uv run python -c "from neural_network.digits import MnistFile; MnistFile('data/mnist.npz').download()"
```

Until it is there, the tests that need it are skipped.

```python
from random import Random

from neural_network.arrays import ArrayBackpropagation
from neural_network.batches import Shuffle
from neural_network.digits import BatchTrainer, Confusion, DigitReader, MnistFile, SoftmaxCrossEntropy
from neural_network.gradient_descent import GradientDescent

file = MnistFile("data/mnist.npz")
training, validation, test = file.training(), file.validation(), file.test()

reader = DigitReader(hidden_neurons=64, random=Random(1))
trainer = BatchTrainer(reader, ArrayBackpropagation(SoftmaxCrossEntropy()), GradientDescent(learning_rate=0.1))
shuffle = Shuffle(Random(101))
for epoch in range(10):
    trainer.epoch(training.inputs, training.truths, batch_size=32, shuffle=shuffle)

reader.right(validation)                                     # 9743 of 10,000
reader.right(test)                                           # 9774 of 10,000
Confusion(test.labels, reader.digits(test.inputs)).worst_mix_up()   # (8, 3, 14): 14 eights read as a 3
```

That run takes about 9 seconds on one laptop. How the reader does on the
validation pile as it trains:

| Steps taken | Pictures seen | Right, of 10,000 |
|-------------|---------------|------------------|
| 0 | 0 | 865 |
| 5 | 160 | 2,711 |
| 15 | 480 | 5,366 |
| 40 | 1,280 | 7,945 |
| 1,563 (1 epoch) | 50,000 | 9,422 |
| 4,689 (3 epochs) | 150,000 | 9,672 |
| 15,630 (10 epochs) | 500,000 | 9,743 |

One weakness is worth knowing. Every pixel has its own weights, so the reader
only knows a stroke in the place where it saw it. Slide every test picture 2
pixels to the right and it reads 8,542 right. At 4 pixels it reads 4,191.

`tests/test_digits.py` checks softmax and its slope by hand and against a
nudge, every kind of slope of the reader against a nudge, and one epoch on the
real pictures.

`tools/export_digit_reader.py` runs the training above and writes the pictures
and weights the article's animations draw.

Uses: `arrays`, `batches`, `initialization`, `training`.

Article: [MNIST digit classifier](https://byteblueprints.com/artificial-intelligence/ann/21_mnist/).

## See it work

```bash
uv run pytest tests/test_digits.py
```

Each test prints its steps under its name, so you can watch the numbers move. For example:

```text
tests/test_digits.py::test_softmax_turns_scores_into_chances_that_add_up_to_1 PASSED
      scores [2.0, 1.0, 0.0]
      e to the power of each [7.3891, 2.7183, 1.0]   together 11.1073
      chances [0.6652, 0.2447, 0.09]   together 1.0

tests/test_digits.py::test_the_slope_of_every_score_is_its_chance_minus_its_truth PASSED
      chances [0.6652, 0.2447, 0.09]   truths [0.0, 1.0, 0.0]
      slopes  [0.6652, -0.7553, 0.09]

tests/test_digits.py::test_one_epoch_reads_most_digits_right PASSED
      right before training: 865 of 10,000 validation pictures
      right after one epoch: 9422   worst mix up (true, read, times) (9, 4, 43)
```
