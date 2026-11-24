from ai_01_perceptron.__main__ import LEARNING_RATE, POOR_START
from ai_01_perceptron.gates import AND, OR, XOR
from ai_01_perceptron.perceptron import Perceptron, Weights
from ai_01_perceptron.trainer import Trainer


def train(gate):
    perceptron = Perceptron(POOR_START, LEARNING_RATE)
    return perceptron, Trainer(max_passes=8).train(perceptron, gate)


def test_hand_set_weights_do_and_and_or():
    for gate, bias in ((AND, -1.5), (OR, -0.5)):
        assert Perceptron(Weights(1, 1, bias)).wrong_corners(gate) == []


def test_first_fix_on_and_matches_the_article():
    _, run = train(AND)
    first = run.fixes[0]
    assert (first.corner.x1, first.corner.x2, first.guess) == (0, 1, 1)
    assert first.weights_after == Weights(w1=-0.5, w2=0.5, bias=-0.75)


def test_and_is_learned_in_four_passes():
    perceptron, run = train(AND)
    assert run.mistakes_per_pass == [2, 2, 1, 0]
    assert perceptron.weights == Weights(w1=0.5, w2=0.5, bias=-0.75)


def test_or_is_learned_in_four_passes():
    perceptron, run = train(OR)
    assert run.mistakes_per_pass == [1, 2, 1, 0]
    assert perceptron.weights == Weights(w1=0.5, w2=1.0, bias=-0.25)


def test_xor_is_never_learned():
    perceptron, run = train(XOR)
    assert not run.learned
    assert len(perceptron.wrong_corners(XOR)) == 2
