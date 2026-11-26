"""Runs every training run the blog's tuning article shows, and writes their numbers.

    uv run python tools/export_tuning_runs.py <folder to write into>

It writes numbers.json: five runs with a different thing to fix in each, a
learning rate range test, a steady rate against warm up and decay, a grid
search against random searches, and the tuned digit reader. Every run uses
the library's own pieces on the MNIST pictures. It takes a few minutes.
"""

import json
import os
import sys
from random import Random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_network.arrays import ArrayBackpropagation  # noqa: E402
from neural_network.batches import Shuffle  # noqa: E402
from neural_network.digits import BatchTrainer, DigitReader, MnistFile, SoftmaxCrossEntropy  # noqa: E402
from neural_network.gradient_descent import GradientDescent  # noqa: E402
from neural_network.tuning import (CurveReading, Dial, GridSearch, RandomSearch, RateRangeTest,  # noqa: E402
                                   ScheduledTrainer, Steady, WarmUpThenDecay)

np.seterr(all="ignore")
BACKPROPAGATION = ArrayBackpropagation(SoftmaxCrossEntropy())
BATCH, SEED = 32, 1
FILE = MnistFile(os.path.join(os.path.dirname(__file__), "..", "data", "mnist.npz"))
TRAINING, VALIDATION, TEST = FILE.training(), FILE.validation(), FILE.test()


def rounded(values) -> list[float]:
    return [round(float(value), 4) for value in values]


def curve(name: str, hidden: int, rate: float, epochs: int, pictures: int) -> dict:
    """One run with a steady rate: both losses after every epoch, and what the library reads from them."""
    reader = DigitReader(hidden, Random(SEED))
    trainer = BatchTrainer(reader, BACKPROPAGATION, GradientDescent(rate))
    losses = trainer.train_and_validate(TRAINING.inputs[:pictures], TRAINING.truths[:pictures], VALIDATION.inputs,
                                        VALIDATION.truths, epochs, BATCH, Shuffle(Random(SEED + 100)))
    says = CurveReading(losses, good_enough=0.1).says()
    print(name, says, reader.right(VALIDATION), flush=True)
    return {"hidden": hidden, "rate": rate, "epochs": epochs, "pictures": pictures, "says": says,
            "training": rounded(losses.training), "validation": rounded(losses.validation), "right": reader.right(VALIDATION)}


def scheduled(hidden: int, schedule, epochs: int) -> dict:
    """One run on all the training pictures with a schedule: the validation loss after every epoch."""
    reader = DigitReader(hidden, Random(SEED))
    trainer = ScheduledTrainer(reader, BACKPROPAGATION, schedule)
    losses = trainer.train_and_validate(TRAINING.inputs, TRAINING.truths, VALIDATION.inputs, VALIDATION.truths,
                                        epochs, BATCH, Shuffle(Random(SEED + 100)))
    print("scheduled", hidden, type(schedule).__name__, reader.right(VALIDATION), flush=True)
    return {"validation": rounded(losses.validation), "training": rounded(losses.training),
            "right": reader.right(VALIDATION), "test": reader.right(TEST)}


def trial_score(trial: dict) -> int:
    """A short run for one trial of a search: 2 epochs on 10,000 pictures. The score is the validation pictures read right."""
    reader = DigitReader(round(trial["hidden_neurons"]), Random(SEED))
    trainer = BatchTrainer(reader, BACKPROPAGATION, GradientDescent(trial["learning_rate"]))
    shuffle = Shuffle(Random(SEED + 100))
    for epoch in range(2):
        trainer.epoch(TRAINING.inputs[:10000], TRAINING.truths[:10000], BATCH, shuffle)
    return reader.right(VALIDATION)


def searched(trials: list[dict]) -> list[dict]:
    return [{"learning_rate": round(trial["learning_rate"], 4), "hidden_neurons": round(trial["hidden_neurons"]),
             "right": trial_score(trial)} for trial in trials]


def main(folder: str) -> None:
    numbers = {"curves": {
        "healthy": curve("healthy", 64, 0.1, 10, 50000),
        "too_high": curve("too high", 64, 2.0, 10, 50000),
        "too_low": curve("too low", 64, 0.001, 10, 50000),
        "underfit": curve("underfit", 2, 0.1, 10, 50000),
        "overfit": curve("overfit", 64, 0.1, 60, 500),
    }}

    test = RateRangeTest(lowest=0.0001, highest=10.0, steps=300)
    test.run(DigitReader(64, Random(SEED)), BACKPROPAGATION, TRAINING.inputs, TRAINING.truths, BATCH, Shuffle(Random(SEED + 100)))
    smoothed = test.smoothed()
    numbers["range"] = {"rates": [float(f"{rate:.4g}") for rate in test.rates], "losses": rounded(test.losses),
                        "smoothed": rounded(smoothed), "lowest": round(test.rates[smoothed.index(min(smoothed))], 4),
                        "suggested": round(test.suggested_rate(), 4)}

    steps = 10 * 1563
    numbers["schedules"] = {"steady": scheduled(64, Steady(0.3), 10),
                            "warm": scheduled(64, WarmUpThenDecay(peak=0.3, warm_up_steps=500, total_steps=steps), 10),
                            "peak": 0.3, "warm_up_steps": 500, "total_steps": steps}

    dials = [Dial("learning_rate", 0.001, 1.0), Dial("hidden_neurons", 16, 256)]
    grid = searched(GridSearch(dials, values_per_dial=3).trials())
    best_of_grid = max(trial["right"] for trial in grid)
    by_seed = []
    for seed in range(1, 13):
        trials = searched(RandomSearch(dials, count=9, random=Random(seed)).trials())
        by_seed.append({"seed": seed, "best": max(trial["right"] for trial in trials), "trials": trials})
        print("random search", seed, by_seed[-1]["best"], "grid", best_of_grid, flush=True)
    numbers["search"] = {"grid": grid, "gridBest": best_of_grid, "random": by_seed[0]["trials"], "randomBest": by_seed[0]["best"],
                         "bestBySeed": [run["best"] for run in by_seed],
                         "randomWins": sum(1 for run in by_seed if run["best"] > best_of_grid)}

    # The tuned reader: more neurons, a higher rate, a little longer. Chosen on the validation pile, tested once.
    steps = 12 * 1563
    steady = scheduled(256, Steady(0.3), 12)
    warm = scheduled(256, WarmUpThenDecay(peak=0.3, warm_up_steps=500, total_steps=steps), 12)
    numbers["tuned"] = {"hidden": 256, "rate": 0.3, "epochs": 12, "steady": steady, "warm": warm,
                        "before": {"hidden": 64, "rate": 0.1, "epochs": 10, "right": 9743, "test": 9774}}
    with open(os.path.join(folder, "numbers.json"), "w", encoding="utf-8") as out:
        json.dump(numbers, out)
        out.write("\n")
    print("written", flush=True)


main(sys.argv[1])
