from random import Random

import pytest
from pytest import approx

np = pytest.importorskip("numpy")

from neural_network.arrays import ArrayBackpropagation, ArrayCrossEntropy, ExampleArrays
from neural_network.batches import Shuffle
from neural_network.digits import BatchTrainer, DigitReader, SoftmaxCrossEntropy
from neural_network.gradient_descent import GradientDescent
from neural_network.training import Losses
from neural_network.tuning import (CurveReading, Dial, GridSearch, GrowingRate, RandomSearch, RateRangeTest,
                                   ScheduledTrainer, Steady, WarmUpThenDecay)

# Five runs, each with a different thing to fix. Training loss first, then validation loss.
RUNS = {
    "learning rate too high": ([2.39, 2.03, 2.02, 2.07, 2.03, 2.03, 2.05, 2.23, 2.20, 2.02, 2.32],
                               [2.40, 2.04, 2.03, 2.08, 2.05, 2.04, 2.10, 2.24, 2.26, 2.05, 2.32]),
    "learning rate too low": ([2.39, 1.65, 1.11, 0.84, 0.69, 0.60, 0.55, 0.51, 0.48, 0.45, 0.43],
                              [2.40, 1.64, 1.09, 0.80, 0.65, 0.56, 0.51, 0.47, 0.44, 0.42, 0.40]),
    "underfitting": ([2.46, 1.25, 1.21, 1.02, 1.07, 1.06, 1.01, 1.08, 1.04, 1.02, 0.99],
                     [2.46, 1.22, 1.17, 0.99, 1.03, 1.04, 0.99, 1.04, 1.00, 1.00, 0.97]),
    "overfitting": ([2.41, 0.97, 0.53, 0.30, 0.18, 0.11, 0.07, 0.05, 0.03, 0.02, 0.01],
                    [2.40, 1.13, 0.77, 0.59, 0.51, 0.49, 0.49, 0.50, 0.52, 0.55, 0.58]),
    "healthy": ([2.39, 0.21, 0.15, 0.11, 0.09, 0.07, 0.08, 0.06, 0.05, 0.05, 0.04],
                [2.40, 0.20, 0.15, 0.12, 0.11, 0.10, 0.11, 0.09, 0.10, 0.09, 0.09]),
}


@pytest.mark.parametrize("wrong", list(RUNS))
def test_the_curves_say_what_to_fix_first(wrong, show):
    training, validation = RUNS[wrong]
    reading = CurveReading(Losses(training, validation), good_enough=0.1)
    show("training loss ends at", training[-1], "  went up in", reading.epochs_the_loss_went_up(), "of 10 epochs")
    show("validation loss was best at epoch", reading.best_epoch(), "  best", validation[reading.best_epoch()], "  ends at", validation[-1])
    show("says:", reading.says())
    assert reading.says() == wrong


def test_a_loss_that_is_not_a_number_means_the_rate_is_too_high(show):
    reading = CurveReading(Losses([2.4, 9.7, float("nan")], [2.4, 9.9, float("nan")]), good_enough=0.1)
    show("training loss", "2.4, 9.7, nan", "  says:", reading.says())
    assert reading.rate_is_too_high()


def test_warm_up_climbs_to_the_peak_then_glides_down(show):
    schedule = WarmUpThenDecay(peak=0.2, warm_up_steps=4, total_steps=12)
    rates = [schedule.rate(step) for step in range(12)]
    show("warm up", rates[:4])
    show("decay  ", rates[4:])
    assert rates[:4] == approx([0.05, 0.1, 0.15, 0.2])
    assert rates[4] == approx(0.2) and rates[8] == approx(0.1)  # half way down the decay, half the peak
    assert all(rates[step] > rates[step + 1] for step in range(4, 11))
    assert 0 < rates[11] < 0.01
    assert Steady(0.1).rate(0) == Steady(0.1).rate(999) == 0.1


def test_a_growing_rate_is_multiplied_by_the_same_number_each_step(show):
    schedule = GrowingRate(lowest=0.001, highest=1.0, steps=4)
    rates = [schedule.rate(step) for step in range(4)]
    show("rates", rates, "  each is", schedule.growth, "times the one before")
    assert rates == approx([0.001, 0.01, 0.1, 1.0])


def test_a_steady_schedule_trains_like_the_plain_batch_trainer(mango_network, new_mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    backpropagation = ArrayBackpropagation(ArrayCrossEntropy())
    plain = BatchTrainer(as_arrays(mango_network), backpropagation, GradientDescent(0.5))
    scheduled = ScheduledTrainer(as_arrays(new_mango_network()), backpropagation, Steady(0.5))
    for trainer in (plain, scheduled):
        for epoch in range(3):
            trainer.epoch(mangoes.inputs, mangoes.truths, batch_size=12, shuffle=Shuffle(Random(epoch)))
    show("steps taken", scheduled.steps_taken, "  loss", scheduled.loss(mangoes.inputs, mangoes.truths),
         "  batch trainer", plain.loss(mangoes.inputs, mangoes.truths))
    assert scheduled.steps_taken == 9
    assert scheduled.loss(mangoes.inputs, mangoes.truths) == approx(plain.loss(mangoes.inputs, mangoes.truths), abs=1e-12)


def test_a_scheduled_trainer_steps_at_the_rate_of_each_step(mango_network, as_arrays, labelled_crate, show):
    mangoes = ExampleArrays(labelled_crate)
    schedule = WarmUpThenDecay(peak=0.4, warm_up_steps=2, total_steps=6)
    trainer = ScheduledTrainer(as_arrays(mango_network), ArrayBackpropagation(ArrayCrossEntropy()), schedule)
    used = []
    for step in range(6):
        trainer.step(mangoes.inputs, mangoes.truths)
        used.append(trainer.gradient_descent.learning_rate)
    show("rates used", used)
    assert used == approx([schedule.rate(step) for step in range(6)])
    assert used[1] == approx(0.4)


def test_a_grid_tries_few_values_of_each_dial_and_random_tries_many(show):
    dials = [Dial("learning_rate", 0.001, 1.0), Dial("hidden_neurons", 16, 256)]
    grid = GridSearch(dials, values_per_dial=3).trials()
    random = RandomSearch(dials, count=9, random=Random(1)).trials()
    for name, trials in [("grid  ", grid), ("random", random)]:
        rates = sorted({round(trial["learning_rate"], 4) for trial in trials})
        show(name, len(trials), "trials,", len(rates), "learning rates:", rates)
    assert len(grid) == len(random) == 9
    assert sorted({trial["learning_rate"] for trial in grid}) == approx([0.001, 0.0316, 1.0], rel=0.001)
    assert sorted({trial["hidden_neurons"] for trial in grid}) == approx([16, 64, 256])
    assert len({trial["learning_rate"] for trial in random}) == 9
    assert all(0.001 <= trial["learning_rate"] <= 1.0 and 16 <= trial["hidden_neurons"] <= 256 for trial in random)
    assert RandomSearch(dials, 9, Random(1)).trials() == random  # the same seed draws the same trials


def test_a_dial_is_as_likely_to_land_low_as_high(show):
    dial = Dial("learning_rate", 0.001, 1.0)
    random = Random(3)
    values = [dial.anywhere(random) for _ in range(3000)]
    below = [sum(1 for value in values if value < edge) for edge in (0.01, 0.1)]
    show("of 3,000 draws:", below[0], "below 0.01,", below[1] - below[0], "from 0.01 to 0.1,", 3000 - below[1], "above 0.1")
    assert dial.evenly(4) == approx([0.001, 0.01, 0.1, 1.0])
    assert below[0] == approx(1000, abs=80) and below[1] == approx(2000, abs=80)


def test_the_range_test_finds_the_rates_where_the_loss_falls(mnist, show):
    training = mnist.training()
    test = RateRangeTest(lowest=0.0001, highest=10.0, steps=300)
    test.run(DigitReader(64, Random(1)), ArrayBackpropagation(SoftmaxCrossEntropy()),
             training.inputs, training.truths, batch_size=32, shuffle=Shuffle(Random(101)))
    smoothed = test.smoothed()
    for step in (0, 100, 150, 200, 220, 250, 299):
        show("step", step, "  rate", test.rates[step], "  loss", smoothed[step])
    show("suggested rate", test.suggested_rate())
    assert len(test.rates) == len(test.losses) == 300
    assert test.rates[0] == approx(0.0001) and test.rates[299] == approx(10.0)
    assert smoothed[100] > 2.0  # at tiny rates nothing is learned in a few steps
    assert min(smoothed) < 1.0  # then the loss falls
    assert smoothed[299] > min(smoothed) * 2  # and at huge rates it blows up
    assert 0.02 < test.suggested_rate() < 0.2
