import random

from pytest import approx

from neural_network.backpropagation import Backpropagation
from neural_network.batches import Shuffle
from neural_network.early_stopping import EarlyStopping
from neural_network.gradient_descent import GradientDescent
from neural_network.loss import SquaredError
from neural_network.network import Network
from neural_network.splits import Split
from neural_network.training import Trainer


def trainer_for(network: Network) -> Trainer:
    return Trainer(network, Backpropagation(SquaredError()), GradientDescent(learning_rate=0.5))


def weights_of(network: Network) -> list:
    return [(neuron.weights, neuron.bias) for layer in network.layers for neuron in layer.neurons]


def test_it_stops_20_epochs_after_the_best_one_and_keeps_the_best(labelled_crate, mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    trainer = trainer_for(mango_network)
    stopping = EarlyStopping(trainer, patience=20)
    losses = stopping.train(split.training, split.validation, epochs=300, batch_size=6, shuffle=Shuffle(random.Random(5)))
    last = len(losses.validation) - 1
    for epoch in (0, 25, 50, stopping.best_epoch, last):
        show("after epoch", epoch, "  training loss", losses.training[epoch], "  validation loss", losses.validation[epoch])
    show("best epoch", stopping.best_epoch, "  stopped after epoch", last, "of 300")
    show("the network that is kept has a validation loss of", trainer.loss(split.validation))
    assert (stopping.best_epoch, last) == (75, 95)
    assert losses.validation[75] == min(losses.validation)
    assert losses.validation[95] > losses.validation[75]  # the last network was worse on the validation set
    assert trainer.loss(split.validation) == approx(losses.validation[75])  # the best network, not the last


def test_the_network_it_keeps_is_the_one_from_the_best_epoch(labelled_crate, mango_network, new_mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    EarlyStopping(trainer_for(mango_network), patience=20).train(split.training, split.validation, epochs=300,
                                                                 batch_size=6, shuffle=Shuffle(random.Random(5)))
    only_75 = new_mango_network()
    trainer_for(only_75).train(split.training, epochs=75, batch_size=6, shuffle=Shuffle(random.Random(5)))
    show("stopped early, kept:  ", weights_of(mango_network)[0])
    show("trained 75 epochs:    ", weights_of(only_75)[0])
    assert weights_of(mango_network) == weights_of(only_75)


def test_the_kept_numbers_are_a_copy_that_later_steps_cannot_change(mango_network, mango, show):
    trainer = trainer_for(mango_network)
    stopping = EarlyStopping(trainer, patience=5)
    kept = stopping.numbers_of(mango_network)
    trainer.step([mango])
    show("kept", kept[0][0].weights, "  after one more step the network has", mango_network.layers[0].neurons[0].weights)
    assert kept[0][0].weights == [0.8, -0.4]
    stopping.put_back(mango_network, kept)
    show("put back:", mango_network.layers[0].neurons[0].weights)
    assert mango_network.layers[0].neurons[0].weights == [0.8, -0.4]
    assert mango_network.layers[1].neurons[0].bias == -0.1


def test_when_training_never_helps_the_validation_set_the_start_is_kept(labelled_crate, mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    wrong_way = [(inputs, [1.0 - truths[0]]) for inputs, truths in split.validation]  # every label turned around
    before = [(list(weights), bias) for weights, bias in weights_of(mango_network)]
    stopping = EarlyStopping(trainer_for(mango_network), patience=5)
    losses = stopping.train(split.training, wrong_way, epochs=300, batch_size=6, shuffle=Shuffle(random.Random(5)))
    show("validation loss after each epoch:", losses.validation)
    show("best epoch", stopping.best_epoch, "  stopped after epoch", len(losses.validation) - 1)
    assert (stopping.best_epoch, len(losses.validation) - 1) == (0, 5)
    assert weights_of(mango_network) == before


def test_with_enough_patience_it_runs_every_epoch(labelled_crate, mango_network, show):
    split = Split(labelled_crate, validation_size=6, test_size=6, shuffle=Shuffle(random.Random(5)))
    stopping = EarlyStopping(trainer_for(mango_network), patience=1000)
    losses = stopping.train(split.training, split.validation, epochs=100, batch_size=6, shuffle=Shuffle(random.Random(5)))
    show("ran", len(losses.validation) - 1, "epochs   best epoch", stopping.best_epoch)
    assert len(losses.validation) - 1 == 100
    assert stopping.best_epoch == 75
