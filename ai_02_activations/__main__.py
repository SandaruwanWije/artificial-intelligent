"""The activation functions article as a program: `uv run python -m ai_02_activations`."""

from pathlib import Path

from .activations import FOUR_CURVES, GELU, NoBend, ReLU, Sigmoid, Step, Tanh
from .bump import BUMP, XS, goal
from .figures import draw_four_curves, draw_step_to_sigmoid
from .figures_bump import draw_no_bend, draw_relu_bump
from .figures_ring import draw_ring
from .layers import Layer, stack
from .network import TinyNetwork
from .report import (final_neuron_line, layer_line, lowest_line, percent_table, point_line,
                     row, slopes_line)
from .ring import ring
from .trainer import Trainer, TrainingRun

IMAGES = Path(__file__).parent / "images"

LAYER_1 = Layer(w=0.5, b=0.5)
LAYER_2 = Layer(w=2, b=-1)
A_HUNDRED_LAYERS = [LAYER_1, LAYER_2] * 50
RING = ring(count=200, noise=0.1, seed=5)
HIDDEN_COUNT = 4
TRAINER = Trainer(steps=2000, learning_rate=0.3, momentum=0.9)
SHOWN_STEPS = (0, 10, 20, 30, 40, 50, 100, 150, 2000)


def from_step_to_curve() -> None:
    print("From step to curve")
    for curve, zs in ((Step(), (-1, 1, 0)), (Sigmoid(), (-2, 0, 2))):
        print(f"  {curve.name}")
        for z in zs:
            print(f"    {point_line(curve, z)}")


def four_curves_and_their_slopes() -> None:
    print("\nFour curves and their slopes")
    for curve in FOUR_CURVES:
        print(f"  {slopes_line(curve, (-3, 0, 3))}")
    print(f"  {lowest_line(GELU(), [z / 100 for z in range(-400, 1)])}")


def why_the_bend_matters() -> None:
    print("\nWhy the bend matters")
    print(f"  {row('x', XS)}")
    print(f"  {row('goal', [goal(x) for x in XS])}")
    print(f"  {layer_line('layer 1', LAYER_1, 2)}")
    print(f"  {layer_line('layer 2', LAYER_2, LAYER_1(2))}")
    print(f"  {layer_line('together', LAYER_1.then(LAYER_2), 2)}")
    print(f"  {layer_line('100 layers', stack(A_HUNDRED_LAYERS), 2)}")


def drawing_it_with_relu() -> None:
    print("\nDrawing it with ReLU")
    n1 = BUMP.inputs[0]
    print(f"  {row('x', XS)}")
    print(f"  {row('z = x', [n1.z(x) for x in XS])}")
    for neuron in BUMP.inputs:
        print(f"  {row(neuron.name, [neuron.gives(x) for x in XS])}")
    print(f"  {row('y', [BUMP.y(x) for x in XS])}")
    print(f"  {final_neuron_line(BUMP)}")


def same_network_only_the_bend() -> list[TrainingRun]:
    print("\nSame network, only the bend")
    runs = [TRAINER.train(TinyNetwork(activation, HIDDEN_COUNT, seed=19), RING)
            for activation in (NoBend(), ReLU(), Tanh())]
    print(f"  {len(RING)} points, {HIDDEN_COUNT} hidden neurons, learning rate "
          f"{TRAINER.learning_rate:g}; % right after each number of steps:")
    for line in percent_table(runs, SHOWN_STEPS):
        print(f"    {line}")
    return runs


def main() -> None:
    from_step_to_curve()
    four_curves_and_their_slopes()
    why_the_bend_matters()
    drawing_it_with_relu()
    runs = same_network_only_the_bend()

    draw_step_to_sigmoid(IMAGES / "step_to_sigmoid.png")
    draw_four_curves(FOUR_CURVES, IMAGES / "four_curves.png")
    draw_no_bend(LAYER_1, LAYER_2, IMAGES / "no_bend.png")
    draw_relu_bump(BUMP, IMAGES / "relu_bump.png")
    draw_ring(RING, runs, IMAGES / "ring.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
