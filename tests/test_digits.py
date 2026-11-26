import math
from random import Random

import pytest
from pytest import approx

np = pytest.importorskip("numpy")

from neural_network.arrays import ArrayBackpropagation
from neural_network.batches import Shuffle
from neural_network.digits import BatchTrainer, Confusion, DigitPictures, DigitReader, Softmax, SoftmaxCrossEntropy
from neural_network.gradient_descent import GradientDescent

LOSS = SoftmaxCrossEntropy()
THREE_SCORES = np.array([[2.0, 1.0, 0.0]])


def test_softmax_turns_scores_into_chances_that_add_up_to_1(show):
    chances = Softmax().chances(THREE_SCORES)
    show("scores", THREE_SCORES[0].tolist())
    show("e to the power of each", [math.exp(2), math.exp(1), math.exp(0)], "  together", math.exp(2) + math.exp(1) + 1)
    show("chances", chances[0].tolist(), "  together", float(chances.sum()))
    assert chances[0].tolist() == approx([0.6652, 0.2447, 0.0900], abs=0.00005)
    assert chances.sum() == approx(1.0)
    # Only the gaps between the scores matter: 100 more on every score gives the same chances.
    assert np.allclose(Softmax().chances(THREE_SCORES + 100), chances)


def test_the_loss_is_the_surprise_at_the_true_class(show):
    for true_class, cost in [(0, 0.4076), (2, 2.4076)]:
        truths = np.eye(3)[[true_class]]
        costs = LOSS.costs(THREE_SCORES, truths)
        show("true class", true_class, "  its chance", float(Softmax().chances(THREE_SCORES)[0, true_class]), "  loss", float(costs[0]))
        assert costs[0] == approx(cost, abs=0.00005)


def test_the_slope_of_every_score_is_its_chance_minus_its_truth(show):
    truths = np.array([[0.0, 1.0, 0.0]])
    slopes = LOSS.slopes(THREE_SCORES, truths)
    show("chances", Softmax().chances(THREE_SCORES)[0].tolist(), "  truths", truths[0].tolist())
    show("slopes ", slopes[0].tolist())
    assert slopes[0].tolist() == approx([0.6652, -0.7553, 0.0900], abs=0.00005)
    # The same slopes the slow way: nudge one score a little and see how far the loss moves.
    for place in range(3):
        nudged = THREE_SCORES.copy()
        nudged[0, place] += 1e-6
        by_nudging = (LOSS.costs(nudged, truths)[0] - LOSS.costs(THREE_SCORES, truths)[0]) / 1e-6
        assert slopes[0, place] == approx(by_nudging, abs=1e-5)


def random_pictures(count: int, seed: int) -> DigitPictures:
    random = Random(seed)
    pixels = [[[random.randrange(256) for _ in range(28)] for _ in range(28)] for _ in range(count)]
    return DigitPictures(np.array(pixels), np.array([random.randrange(10) for _ in range(count)]))


def test_a_picture_becomes_one_row_of_784_numbers(show):
    pictures = np.zeros((2, 28, 28), dtype=int)
    pictures[0, 2, 5] = 255  # full ink in row 2, column 5 of the first picture
    pictures[1, 27, 27] = 51
    digits = DigitPictures(pictures, np.array([3, 0]))
    show("inputs", digits.inputs.shape, "  truths", digits.truths.shape)
    show("row 2, column 5 is place 2 x 28 + 5 =", 2 * 28 + 5, "  its input", float(digits.inputs[0, 61]))
    show("truth for a 3", digits.truths[0].tolist())
    assert (digits.inputs.shape, digits.truths.shape) == ((2, 784), (2, 10))
    assert digits.inputs[0, 61] == 1.0 and digits.inputs[0].sum() == 1.0
    assert digits.inputs[1, 783] == approx(0.2)
    assert digits.truths.tolist() == [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def test_a_new_reader_only_guesses(show):
    reader = DigitReader(hidden_neurons=64, random=Random(1))
    hidden, scores = reader.layers
    pictures = random_pictures(3, seed=7)
    chances = reader.chances(pictures.inputs)
    show("hidden weights", hidden.weights.shape, "  score weights", scores.weights.shape)
    show("numbers to learn", hidden.weights.size + hidden.biases.size + scores.weights.size + scores.biases.size)
    show("chances for one picture", chances[0].tolist())
    assert (hidden.weights.shape, scores.weights.shape) == ((784, 64), (64, 10))
    assert hidden.weights.size + hidden.biases.size + scores.weights.size + scores.biases.size == 50890
    assert chances.shape == (3, 10) and np.allclose(chances.sum(axis=1), 1.0)
    assert reader.digits(pictures.inputs).tolist() == chances.argmax(axis=1).tolist()


def test_every_slope_matches_a_tiny_nudge(show):
    reader = DigitReader(hidden_neurons=5, random=Random(2))
    pictures = random_pictures(4, seed=3)
    trainer = BatchTrainer(reader, ArrayBackpropagation(LOSS), GradientDescent(0.1))
    slopes = ArrayBackpropagation(LOSS).slopes(reader, pictures.inputs, pictures.truths)
    before = trainer.loss(pictures.inputs, pictures.truths)
    for layer_number, row, column in [(0, 100, 0), (0, 400, 3), (0, 783, 4), (1, 0, 0), (1, 2, 7), (1, 4, 9)]:
        weights = reader.layers[layer_number].weights
        weights[row, column] += 1e-6
        by_nudging = (trainer.loss(pictures.inputs, pictures.truths) - before) / 1e-6
        weights[row, column] -= 1e-6
        show("layer", layer_number, "weight", (row, column), "  slope", float(slopes[layer_number].weights[row, column]), "  by nudging", by_nudging)
        assert slopes[layer_number].weights[row, column] == approx(by_nudging, abs=1e-5)
    assert slopes[0].weights.shape == (784, 5) and slopes[1].weights.shape == (5, 10)


class CountingTrainer(BatchTrainer):
    """Keeps the size of every batch it steps on."""

    sizes: list[int] = []

    def step(self, inputs, truths):
        self.sizes.append(inputs.shape[0])
        super().step(inputs, truths)


def test_an_epoch_is_one_step_for_every_batch(show):
    pictures = random_pictures(10, seed=5)
    trainer = CountingTrainer(DigitReader(4, Random(1)), ArrayBackpropagation(LOSS), GradientDescent(0.05))
    before = trainer.loss(pictures.inputs, pictures.truths)
    trainer.epoch(pictures.inputs, pictures.truths, batch_size=4, shuffle=Shuffle(Random(1)))
    after = trainer.loss(pictures.inputs, pictures.truths)
    show("10 pictures in batches of 4: steps on", trainer.sizes, "pictures")
    show("loss before", before, "  after one epoch", after)
    assert trainer.sizes == [4, 4, 2]
    assert after < before


def test_the_confusion_grid_counts_each_pair(show):
    confusion = Confusion(labels=np.array([4, 4, 4, 9, 7, 7]), read=np.array([4, 9, 9, 9, 7, 1]))
    show("4s read as 4:", int(confusion.counts[4, 4]), "  4s read as 9:", int(confusion.counts[4, 9]))
    show("right", confusion.right(), "of 6", "  worst mix up (true, read, times)", confusion.worst_mix_up())
    assert confusion.counts[4].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 2]
    assert confusion.right() == 3
    assert confusion.worst_mix_up() == (4, 9, 2)


def test_the_file_is_cut_into_three_piles(mnist, show):
    training, validation, test = mnist.training(), mnist.validation(), mnist.test()
    show("training", training.inputs.shape, "  validation", validation.inputs.shape, "  test", test.inputs.shape)
    show("first five labels", training.labels[:5].tolist(), "  pixels with ink in the first picture", int((training.inputs[0] > 0).sum()))
    assert (training.inputs.shape, validation.inputs.shape, test.inputs.shape) == ((50000, 784), (10000, 784), (10000, 784))
    assert training.labels[:5].tolist() == [5, 0, 4, 1, 9]
    assert training.inputs.min() == 0.0 and training.inputs.max() == 1.0


def test_one_epoch_reads_most_digits_right(mnist, show):
    training, validation = mnist.training(), mnist.validation()
    reader = DigitReader(hidden_neurons=64, random=Random(1))
    trainer = BatchTrainer(reader, ArrayBackpropagation(LOSS), GradientDescent(learning_rate=0.1))
    before = reader.right(validation)
    trainer.epoch(training.inputs, training.truths, batch_size=32, shuffle=Shuffle(Random(101)))
    confusion = Confusion(validation.labels, reader.digits(validation.inputs))
    show("right before training:", before, "of 10,000 validation pictures")
    show("right after one epoch:", confusion.right(), "  worst mix up (true, read, times)", confusion.worst_mix_up())
    assert before < 2000  # ten digits to pick from, so a new reader is right about 1 time in 10
    assert confusion.right() == reader.right(validation) > 9000
    assert confusion.counts.sum() == 10000
