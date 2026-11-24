"""Run with: uv run python -m ai_01_perceptron"""

from pathlib import Path

from .figures import draw_boundaries, draw_mistakes, draw_passes
from .gates import AND, OR, XOR, LogicGate
from .perceptron import Perceptron, Weights
from .report import fix_line, summary_lines, z_line
from .trainer import Trainer, TrainingRun

IMAGES = Path(__file__).parent / "images"

POOR_START = Weights(w1=-0.5, w2=1.0, bias=-0.25)
LEARNING_RATE = 0.5
TRAINER = Trainer(max_passes=8)


def one_neuron_with_a_hard_edge() -> None:
    print("One neuron with a hard edge")
    for gate, bias in ((AND, -1.5), (OR, -0.5)):
        neuron = Perceptron(Weights(w1=1, w2=1, bias=bias))
        print(f"  {gate.name}: {neuron.weights}")
        for corner in gate.corners:
            print(f"    {z_line(neuron, corner)}")


def learn(gate: LogicGate, show_fixes: bool = False) -> tuple[Perceptron, TrainingRun]:
    perceptron = Perceptron(POOR_START, LEARNING_RATE)
    run = TRAINER.train(perceptron, gate)
    if show_fixes:
        for fix in run.fixes:
            print(f"  {fix_line(fix)}")
    for line in summary_lines(perceptron, gate, run):
        print(f"  {line}")
    return perceptron, run


def main() -> None:
    one_neuron_with_a_hard_edge()

    print("\nLearning AND")
    and_perceptron, and_run = learn(AND, show_fixes=True)
    print("\nPass after pass: OR")
    or_perceptron, or_run = learn(OR)
    print("\nThe limit: XOR")
    xor_perceptron, xor_run = learn(XOR)

    draw_passes(AND, and_run, IMAGES / "and_passes.png")
    draw_mistakes({AND: and_run, OR: or_run, XOR: xor_run}, IMAGES / "mistakes.png")
    final = {AND: and_perceptron.weights, OR: or_perceptron.weights, XOR: xor_perceptron.weights}
    draw_boundaries(final, IMAGES / "boundaries.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
