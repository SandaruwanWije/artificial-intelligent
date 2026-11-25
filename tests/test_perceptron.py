from neural_network.activations import STEP
from neural_network.neuron import Neuron
from neural_network.perceptron import Perceptron

CORNERS = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND = list(zip(CORNERS, [0, 0, 0, 1]))
OR = list(zip(CORNERS, [0, 1, 1, 1]))
XOR = list(zip(CORNERS, [0, 1, 1, 0]))


def poor_start() -> Perceptron:
    return Perceptron(weights=[-0.5, 1.0], bias=-0.25, learning_rate=0.5)


def test_a_perceptron_is_a_neuron_with_the_step(show):
    perceptron = poor_start()
    show("weights", perceptron.weights, "bias", perceptron.bias, "activation", type(perceptron.activation).__name__)
    show("the step turns a total of -0.3 into", STEP(-0.3), "  and a total of 0.2 into", STEP(0.2))
    assert isinstance(perceptron, Neuron)
    assert perceptron.activation is STEP


def test_weights_set_by_hand_give_and(show):
    perceptron = Perceptron(weights=[1, 1], bias=-1.5)
    for inputs, answer in AND:
        total = perceptron.total(inputs)
        guess = perceptron.guess(inputs)
        show(inputs, "total", total, "-> step ->", guess, "  want", answer)
        assert guess == answer


def test_one_wrong_guess_moves_every_number(show):
    perceptron = poor_start()
    inputs, answer = [0, 1], 0
    show("before: weights", perceptron.weights, "bias", perceptron.bias)
    show("inputs", inputs, "total", perceptron.total(inputs), "guess", perceptron.guess(inputs), "want", answer)
    error = perceptron.learn(inputs, answer)
    show("error = want - guess =", error, "  nudge = 0.5 x error =", perceptron.learning_rate * error)
    show("each weight moves by nudge x its input, the bias by the nudge")
    show("after:  weights", perceptron.weights, "bias", perceptron.bias)
    assert error == -1
    assert (perceptron.weights, perceptron.bias) == ([-0.5, 0.5], -0.75)


def test_and_is_learned_pass_by_pass(show):
    perceptron = poor_start()
    mistakes_per_pass = []
    for pass_number in range(1, 5):
        mistakes = perceptron.train(AND, max_passes=1)[0]
        mistakes_per_pass.append(mistakes)
        show("pass", pass_number, "mistakes", mistakes, "-> weights", perceptron.weights, "bias", perceptron.bias)
    assert mistakes_per_pass == [2, 2, 1, 0]
    assert (perceptron.weights, perceptron.bias) == ([0.5, 0.5], -0.75)


def test_train_stops_after_a_pass_with_no_mistakes(show):
    perceptron = poor_start()
    mistakes_per_pass = perceptron.train(OR)
    show("OR, mistakes in each pass:", mistakes_per_pass)
    for inputs, answer in OR:
        show(inputs, "guess", perceptron.guess(inputs), "want", answer)
    assert mistakes_per_pass == [1, 2, 1, 0]


def test_xor_is_never_learned(show):
    perceptron = poor_start()
    mistakes_per_pass = perceptron.train(XOR)
    show("XOR, mistakes in each of the 8 passes:", mistakes_per_pass)
    show("one straight line cannot put [0, 1] and [1, 0] on one side and the others on the other")
    assert mistakes_per_pass[-1] > 0
