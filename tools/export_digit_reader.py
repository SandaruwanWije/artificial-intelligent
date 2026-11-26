"""Trains the digit reader once and writes what the blog's digit article draws from.

    uv run python tools/export_digit_reader.py <folder to write into>

Two files are written:
  reader.bin    93 real pictures, and the reader's weights at ten moments of its training
  numbers.json  every count and loss the article quotes, from this same run

The layout of reader.bin, little endian:
  "DGT1", then four whole numbers of 2 bytes: hidden neurons, moments, pictures, 0
  each picture: 1 byte for the digit it shows, then its 784 pixels, 1 byte each
  each moment:  the step (4 bytes), then one scale per hidden neuron (4 byte floats),
                each hidden neuron's 784 weights as 1 byte each (weight = byte x scale),
                the hidden biases, each hidden neuron's 10 score weights, and the
                10 score biases (4 byte floats)
"""

import json
import os
import struct
import sys
from random import Random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_network.arrays import ArrayBackpropagation  # noqa: E402
from neural_network.batches import Shuffle  # noqa: E402
from neural_network.digits import BatchTrainer, Confusion, DigitReader, MnistFile, SoftmaxCrossEntropy  # noqa: E402
from neural_network.gradient_descent import GradientDescent  # noqa: E402

HIDDEN, RATE, BATCH, EPOCHS, SEED = 64, 0.1, 32, 10, 1
STEPS_PER_EPOCH = 1563
MOMENTS = [0, 5, 15, 40, 100, 250, 600, STEPS_PER_EPOCH, 3 * STEPS_PER_EPOCH, EPOCHS * STEPS_PER_EPOCH]


def packed(reader: DigitReader, step: int) -> bytes:
    """One moment of the reader, with the hidden weights squeezed into 1 byte each."""
    hidden, scores = reader.layers
    scales = np.abs(hidden.weights).max(axis=0) / 127
    as_bytes = np.round(hidden.weights / scales).astype(np.int8)
    parts = [struct.pack("<I", step), scales.astype("<f4").tobytes(), as_bytes.T.tobytes(),
             hidden.biases.astype("<f4").tobytes(), scores.weights.astype("<f4").tobytes(),
             scores.biases.astype("<f4").tobytes()]
    return b"".join(parts)


class WatchedTrainer(BatchTrainer):
    """A BatchTrainer that keeps the reader, and how it does on the validation pile, at the chosen moments."""

    def watch(self, validation) -> None:
        self.validation, self.steps, self.kept, self.seen = validation, 0, [], []
        self.keep()

    def keep(self) -> None:
        if self.steps in MOMENTS:
            self.kept.append(packed(self.network, self.steps))
            loss = self.loss(self.validation.inputs, self.validation.truths)
            self.seen.append({"step": self.steps, "right": self.network.right(self.validation), "loss": round(loss, 4)})

    def step(self, inputs, truths) -> None:
        super().step(inputs, truths)
        self.steps += 1
        self.keep()


def shifted(pictures, by: int) -> np.ndarray:
    """Every picture moved `by` pixels to the right, with paper coming in on the left."""
    grids = pictures.inputs.reshape(-1, 28, 28)
    moved = np.zeros(grids.shape)
    moved[:, :, by:] = grids[:, :, :28 - by] if by else grids
    return moved.reshape(-1, 784)


def main(folder: str) -> None:
    file = MnistFile(os.path.join(os.path.dirname(__file__), "..", "data", "mnist.npz"))
    training, validation, test = file.training(), file.validation(), file.test()
    reader = DigitReader(HIDDEN, Random(SEED))
    trainer = WatchedTrainer(reader, ArrayBackpropagation(SoftmaxCrossEntropy()), GradientDescent(RATE))
    trainer.watch(validation)
    shuffle = Shuffle(Random(SEED + 100))
    epochs = {"training": [trainer.loss(training.inputs, training.truths)],
              "validation": [trainer.loss(validation.inputs, validation.truths)], "right": [reader.right(validation)]}
    for epoch in range(EPOCHS):
        trainer.epoch(training.inputs, training.truths, BATCH, shuffle)
        epochs["training"].append(trainer.loss(training.inputs, training.truths))
        epochs["validation"].append(trainer.loss(validation.inputs, validation.truths))
        epochs["right"].append(reader.right(validation))
    epochs["training"] = [round(loss, 4) for loss in epochs["training"]]
    epochs["validation"] = [round(loss, 4) for loss in epochs["validation"]]

    chances = reader.chances(test.inputs)
    read = chances.argmax(axis=1)
    sure = chances.max(axis=1)
    confusion = Confusion(test.labels, read)
    # The pictures the article shows: a wall of training digits, two right ones of each digit,
    # one the reader is right about but unsure of, and the twelve mistakes it was surest of.
    wall = [(training.inputs[i], int(training.labels[i])) for i in range(60)]
    feed_rows = [int(np.flatnonzero((test.labels == digit) & (read == digit))[k]) for k in range(2) for digit in range(10)]
    unsure_row = int(np.flatnonzero((read == test.labels) & (sure > 0.55) & (sure < 0.7))[0])
    wrong_rows = np.flatnonzero(read != test.labels)
    mistake_rows = [int(row) for row in wrong_rows[np.argsort(-sure[wrong_rows])][:12]]
    rows = feed_rows + [unsure_row] + mistake_rows
    pictures = wall + [(test.inputs[row], int(test.labels[row])) for row in rows]

    with open(os.path.join(folder, "reader.bin"), "wb") as out:
        out.write(b"DGT1" + struct.pack("<4H", HIDDEN, len(trainer.kept), len(pictures), 0))
        for pixels, digit in pictures:
            out.write(bytes([digit]) + np.round(pixels * 255).astype(np.uint8).tobytes())
        for moment in trainer.kept:
            out.write(moment)

    numbers = {
        "hidden": HIDDEN, "rate": RATE, "batch": BATCH, "epochs": EPOCHS, "stepsPerEpoch": STEPS_PER_EPOCH,
        "toLearn": 784 * HIDDEN + HIDDEN + HIDDEN * 10 + 10,
        "moments": trainer.seen, "byEpoch": epochs,
        "test": {"right": confusion.right(), "of": int(confusion.counts.sum())},
        "confusion": confusion.counts.tolist(), "worst": list(confusion.worst_mix_up()),
        "groups": {"wall": [0, 60], "feed": [60, 80], "unsure": 80, "mistakes": [81, 93]},
        "unsure": {"truth": int(test.labels[unsure_row]), "scores": [round(float(s), 2) for s in reader.answer(test.inputs[[unsure_row]])[0]],
                   "chances": [round(float(c), 4) for c in chances[unsure_row]]},
        "mistakes": [{"truth": int(test.labels[row]), "read": int(read[row]), "chance": round(float(sure[row]), 4)} for row in mistake_rows],
        "shifted": [{"by": by, "right": int((reader.digits(shifted(test, by)) == test.labels).sum())} for by in range(5)],
        "seven": {"chance": round(float(chances[feed_rows[7], 7]), 4),
                  "firing": int((reader.layers[0].outputs(test.inputs[[feed_rows[7]]]) > 0).sum())},
    }
    with open(os.path.join(folder, "numbers.json"), "w", encoding="utf-8") as out:
        json.dump(numbers, out, indent="\t")
        out.write("\n")
    print(json.dumps({key: numbers[key] for key in ["test", "worst", "seven", "shifted"]}))
    print("confusion", numbers["confusion"])


main(sys.argv[1])
