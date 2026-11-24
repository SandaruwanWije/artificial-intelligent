"""The perceptron article, as code: one function per chapter.

Run from the repo root:
    uv run python ai_01_perceptron/main.py
"""

from pathlib import Path

from perceptron import Perceptron
from plots import plot_boundaries, plot_mistakes

IMAGES_DIR = Path(__file__).parent / "images"

# Each gate as its four corners: (x1, x2, the right answer)
GATES = {
    "AND": [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],
    "OR": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
    "XOR": [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
}


def poor_start():
    """The poor first line every gate learns from. On AND it gets (0, 1) wrong."""
    return Perceptron(w1=-0.5, w2=1.0, bias=-0.25, rate=0.5)


def one_neuron_with_a_hard_edge():
    """Weights set by hand: a bias of -1.5 does AND, -0.5 does OR."""
    print("One neuron with a hard edge")
    for gate, bias in [("AND", -1.5), ("OR", -0.5)]:
        neuron = Perceptron(w1=1, w2=1, bias=bias)
        print(f"  {gate}: {neuron}")
        for x1, x2, answer in GATES[gate]:
            z = neuron.total(x1, x2)
            print(f"    ({x1}, {x2})  z = {z:g}  answer {neuron.guess(x1, x2)}")


def learn(gate, show_fixes=False):
    """Teach a fresh perceptron one gate and print how it went."""
    neuron = poor_start()
    mistakes_per_pass = neuron.train(GATES[gate], show_fixes=show_fixes)
    print(f"  mistakes per pass: {', '.join(map(str, mistakes_per_pass))}")
    print(f"  ends on: {neuron}")
    wrong = [(x1, x2) for x1, x2, answer in GATES[gate] if neuron.guess(x1, x2) != answer]
    if not wrong:
        print(f"  learned in {len(mistakes_per_pass)} passes")
    else:
        print(f"  still wrong on {wrong} after {len(mistakes_per_pass)} passes")
    return neuron, mistakes_per_pass


def main():
    one_neuron_with_a_hard_edge()

    print("\nLearning AND")
    and_neuron, and_mistakes = learn("AND", show_fixes=True)

    print("\nPass after pass: OR")
    or_neuron, or_mistakes = learn("OR")

    print("\nThe limit: XOR")
    xor_neuron, xor_mistakes = learn("XOR")

    IMAGES_DIR.mkdir(exist_ok=True)
    neurons = {"AND": and_neuron, "OR": or_neuron, "XOR": xor_neuron}
    plot_boundaries(neurons, GATES, IMAGES_DIR / "boundaries.png")
    mistakes = {"AND": and_mistakes, "OR": or_mistakes, "XOR": xor_mistakes}
    plot_mistakes(mistakes, IMAGES_DIR / "mistakes.png")
    print(f"\nPlots saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
