"""The layers and depth article as a program: one function per chapter, then the pictures."""

from pathlib import Path

from .cloud import xor_cloud
from .figures import draw_new_numbers, draw_one_line, draw_tents
from .figures_cloud import draw_cloud
from .report import band_line, corner_line, depth_table, neuron_line, opened_up, pair, trial_line, try_line
from .tents import DeepOrWide, deep_tents
from .trainer import Trainer, Trial
from .xor import (CORNERS, H1, H2, LINES_ONE_NEURON_TRIES, XOR_NETWORK, Y, band_answer,
                  corners_right, every_line_on_a_grid, wrong_corners)

IMAGES = Path(__file__).parent / "images"

CLOUD = xor_cloud(count=160, spread=0.22, seed=9)
TRAINER = Trainer(epochs=2000, learning_rate=0.3)
SHARED_START, OTHER_START = 8, 7
TRIED = ((1, SHARED_START), (2, SHARED_START), (2, OTHER_START),
         (3, SHARED_START), (4, SHARED_START), (8, SHARED_START))


def one_line_is_not_enough() -> None:
    print("One line is not enough")
    for neuron in LINES_ONE_NEURON_TRIES:
        print(f"  {try_line(neuron, wrong_corners(neuron))}")
    every_line = every_line_on_a_grid()
    best = max(corners_right(neuron) for neuron in every_line)
    print(f"  the best of {len(every_line)} lines: {best} of 4 right")
    print("  two lines, x1 + x2 = 0.5 and x1 + x2 = 1.5:")
    for corner in CORNERS:
        print(f"    {band_line(corner, band_answer(corner))}")


def the_hidden_layer() -> None:
    print("\nThe hidden layer")
    for name, neuron in (("h1", H1), ("h2", H2), ("y", Y)):
        print(f"  {neuron_line(name, neuron)}")
    print("  at each corner:")
    for corner in CORNERS:
        new_numbers = XOR_NETWORK.new_numbers(corner.inputs)
        print(f"    {corner_line(corner, new_numbers, XOR_NETWORK.answer(corner.inputs))}")
    print("  each corner's new numbers (h1, h2):")
    for corner in CORNERS:
        print(f"    {pair(corner.inputs)} -> {pair(XOR_NETWORK.new_numbers(corner.inputs))}")


def real_data() -> list[Trial]:
    print("\nReal data")
    wanting_one = sum(point.wants for point in CLOUD)
    print(f"  {len(CLOUD)} points: {wanting_one} want 1, {len(CLOUD) - wanting_one} want 0")
    trials = [Trial(n, seed, TRAINER.train(CLOUD, n, seed)) for n, seed in TRIED]
    for trial in trials:
        print(f"  {trial_line(trial, CLOUD)}")
        for neuron in trial.switched_off():
            print(f"    {neuron_line('switched off', neuron)}")
    print("  the network with 3 hidden neurons, opened up:")
    for line in opened_up(trials[3]):
        print(f"    {line}")
    return trials


def wider_or_deeper() -> list[DeepOrWide]:
    print("\nWider or deeper")
    spots = (0, 0.25, 0.5, 0.75, 1)
    for layers in (1, 2):
        answers = "  ".join(f"{deep_tents(layers).answer((x,)):g}" for x in spots)
        print(f"  {layers} layer{'s' if layers > 1 else ''} at x = 0, 0.25, 0.5, 0.75, 1:  {answers}")
    rows = [DeepOrWide.for_layers(layers) for layers in (1, 2, 3, 4, 10)]
    for line in depth_table(rows):
        print(f"  {line}")
    return rows


def main() -> None:
    one_line_is_not_enough()
    the_hidden_layer()
    trials = real_data()
    rows = wider_or_deeper()

    draw_one_line(LINES_ONE_NEURON_TRIES, XOR_NETWORK, IMAGES / "one_line.png")
    draw_new_numbers(XOR_NETWORK, IMAGES / "new_numbers.png")
    draw_cloud(trials[:3], CLOUD, IMAGES / "cloud_two.png")
    draw_cloud(trials[3:], CLOUD, IMAGES / "cloud_more.png")
    draw_tents(rows[:4], IMAGES / "tents.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
