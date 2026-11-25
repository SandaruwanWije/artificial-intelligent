import random

from pytest import approx

from neural_network.backpropagation import Backpropagation
from neural_network.batches import Shuffle
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.network import Network
from neural_network.splits import CrossValidation, Split
from neural_network.training import Trainer


def ripe_in(pile: list) -> int:
    return len([truths for _, truths in pile if truths[0] == 1])


def right_in(network: Network, pile: list) -> int:
    """How many mangoes the network calls right: 0.5 or more means ripe."""
    return len([inputs for inputs, truths in pile if (network.answer(inputs)[0] >= 0.5) == (truths[0] == 1)])


def trainer_for(network: Network) -> Trainer:
    return Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))


def test_36_mangoes_are_cut_into_piles_of_24_6_and_6(labelled_crate, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    show("training", len(split.training), "  validation", len(split.validation), "  test", len(split.test))
    show("ripe in each pile:", ripe_in(split.training), ripe_in(split.validation), ripe_in(split.test))
    assert (len(split.training), len(split.validation), len(split.test)) == (24, 6, 6)
    assert (ripe_in(split.training), ripe_in(split.validation), ripe_in(split.test)) == (13, 3, 2)


def test_every_mango_is_in_exactly_one_pile_and_the_same_seed_gives_the_same_piles(labelled_crate, show):
    kept = list(labelled_crate)
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    again = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    piles = split.training + split.validation + split.test
    show("the validation pile:", [inputs for inputs, _ in split.validation])
    assert sorted(piles) == sorted(labelled_crate)
    assert labelled_crate == kept  # the crate keeps its order
    assert (again.training, again.validation, again.test) == (split.training, split.validation, split.test)


def test_a_crate_packed_ripe_first_shows_why_the_shuffle_comes_first(labelled_crate, show):
    packed = sorted(labelled_crate, key=lambda mango: -mango[1][0])
    as_packed = (ripe_in(packed[:24]), ripe_in(packed[24:30]), ripe_in(packed[30:]))
    split = Split(packed, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(2)))
    shuffled = (ripe_in(split.training), ripe_in(split.validation), ripe_in(split.test))
    show("cut as packed: ripe in training, validation, test =", as_packed)
    show("shuffled, then cut:                               ", shuffled)
    assert as_packed == (18, 0, 0)  # the validation and test piles hold no ripe mango at all
    assert shuffled == (13, 2, 3)


def test_30_mangoes_make_5_folds_and_each_fold_validates_once(labelled_crate, show):
    cross = CrossValidation(labelled_crate[:30], folds=5)
    validated = []
    for run in range(5):
        training, validation = cross.training(run), cross.validation(run)
        show("run", run, "  validates on fold", run, "(", len(validation), "mangoes)   trains on", len(training))
        assert (len(training), len(validation)) == (24, 6)
        assert [mango for mango in validation if mango in training] == []
        validated = validated + validation
    assert validated == labelled_crate[:30]


def test_folds_that_cannot_be_equal_differ_by_one_at_most(show):
    cross = CrossValidation(list(range(32)), folds=5)
    show("32 examples in 5 folds:", cross.folds)
    assert [len(fold) for fold in cross.folds] == [6, 6, 6, 7, 7]
    assert cross.training(0) == list(range(6, 32))


def test_training_keeps_a_loss_for_both_sets_after_each_epoch(labelled_crate, mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    trainer = trainer_for(mango_network)
    before = trainer.loss(split.validation)
    losses = trainer.train_and_validate(split.training, split.validation, epochs=200, batch_size=6,
                                        shuffle=Shuffle(random.Random(5)))
    for epoch in (0, 1, 10, 50, 100, 200):
        show("after epoch", epoch, "  training loss", losses.training[epoch], "  validation loss",
             losses.validation[epoch])
    show("right: training", right_in(mango_network, split.training), "of 24   validation",
         right_in(mango_network, split.validation), "of 6")
    show("the test pile, opened once at the end: loss", trainer.loss(split.test), "  right",
         right_in(mango_network, split.test), "of 6")
    assert (len(losses.training), len(losses.validation)) == (201, 201)
    assert losses.validation[0] == before
    assert [round(losses.training[epoch], 4) for epoch in (0, 10, 200)] == [0.2624, 0.1954, 0.0994]
    assert [round(losses.validation[epoch], 4) for epoch in (0, 10, 200)] == [0.2446, 0.1793, 0.1224]
    assert (right_in(mango_network, split.training), right_in(mango_network, split.validation)) == (20, 5)
    assert (round(trainer.loss(split.test), 4), right_in(mango_network, split.test)) == (0.165, 4)


def test_five_runs_grade_the_network_on_30_mangoes_instead_of_6(labelled_crate, new_mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    cross = CrossValidation(split.training + split.validation, folds=5)  # the test pile stays locked
    right, last_loss = [], []
    for run in range(5):
        network = new_mango_network()  # every run trains a new network from the start
        losses = trainer_for(network).train_and_validate(cross.training(run), cross.validation(run), epochs=200,
                                                         batch_size=6, shuffle=Shuffle(random.Random(5)))
        right.append(right_in(network, cross.validation(run)))
        last_loss.append(losses.validation[200])
        show("run", run, "  validation loss", losses.validation[200], "  right", right[run], "of 6")
    show("all five runs together:", right[0] + right[1] + right[2] + right[3] + right[4], "of 30")
    assert right == [3, 4, 5, 6, 5]
    assert last_loss[4] == approx(0.1224, abs=0.0001)  # the last fold is the validation pile from the split
