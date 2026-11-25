import math
import random

from pytest import approx

from neural_network.augmentation import Augmentation
from neural_network.backpropagation import Backpropagation
from neural_network.batches import Batches, Shuffle
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.splits import Split
from neural_network.training import Trainer


def test_24_mangoes_and_3_nudged_copies_of_each_are_96_examples(labelled_crate, show):
    mangoes = labelled_crate[:24]
    more = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(mangoes)
    show(len(mangoes), "mangoes  ->", len(more), "examples")
    show("the first mango", mangoes[0], "  its three copies", more[24], more[48], more[72])
    assert len(more) == 24 + 3 * 24
    assert more[:24] == mangoes  # the real mangoes come first, as they were


def test_a_copy_keeps_the_truth_and_moves_each_input_a_little(labelled_crate, show):
    mangoes = labelled_crate[:24]
    more = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(mangoes)
    moves = []
    for copy in range(24, 96):
        inputs, truths = more[copy]
        real_inputs, real_truths = mangoes[copy % 24]
        assert truths == real_truths  # a nudged ripe mango is still called ripe
        moves.append(inputs[0] - real_inputs[0])
        moves.append(inputs[1] - real_inputs[1])
    usual_move = math.sqrt(sum(move * move for move in moves) / len(moves))
    show("the first six moves:", moves[:6])
    show("the usual size of all", len(moves), "moves is", usual_move, "  the nudge asked for is 0.15")
    assert usual_move == approx(0.15, rel=0.15)
    assert 0.0 not in moves  # no copy is exactly the mango it came from


def test_the_real_mangoes_are_not_changed_and_the_same_seed_gives_the_same_copies(labelled_crate, show):
    kept = [(list(inputs), list(truths)) for inputs, truths in labelled_crate]
    first = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(labelled_crate)
    again = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(labelled_crate)
    none = Augmentation(copies=0, nudge=0.15, random=random.Random(31)).augmented(labelled_crate)
    show("36 mangoes ->", len(first), "examples, and the 36 are as they were. With 0 copies:", len(none))
    assert labelled_crate == kept
    assert first == again
    assert none == labelled_crate


def test_training_on_the_copies_takes_four_times_the_steps_in_every_epoch(labelled_crate, new_mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    more = Augmentation(copies=3, nudge=0.15, random=random.Random(31)).augmented(split.training)
    ends = []
    for name, examples in (("24 real mangoes", split.training), ("96 with the copies", more)):
        network = new_mango_network()
        trainer = Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))
        losses = trainer.train_and_validate(examples, split.validation, epochs=50, batch_size=6,
                                            shuffle=Shuffle(random.Random(5)))
        steps = len(Batches(6).cut(examples))
        show(name, " ", steps, "steps in an epoch   loss on the 24 real mangoes", trainer.loss(split.training),
             "  validation loss", losses.validation[50])
        ends.append((steps, losses.validation[50] < losses.validation[0]))
    assert ends == [(4, True), (16, True)]
