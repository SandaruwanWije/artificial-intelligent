from .gates import Corner, LogicGate
from .perceptron import Perceptron
from .trainer import Fix, TrainingRun


def z_line(neuron: Perceptron, corner: Corner) -> str:
    z = neuron.total(corner.x1, corner.x2)
    return f"({corner.x1}, {corner.x2})  z = {z:g}  answer {neuron.guess(corner.x1, corner.x2)}"


def fix_line(fix: Fix) -> str:
    corner = fix.corner
    return (f"pass {fix.pass_number}  ({corner.x1}, {corner.x2})  guessed {fix.guess}, "
            f"wants {corner.answer}  ->  {fix.weights_after}")


def summary_lines(perceptron: Perceptron, gate: LogicGate, run: TrainingRun) -> list[str]:
    lines = [
        f"mistakes per pass: {', '.join(map(str, run.mistakes_per_pass))}",
        f"ends on: {perceptron.weights}",
    ]
    if run.learned:
        lines.append(f"learned in {run.passes} passes")
    else:
        wrong = [(corner.x1, corner.x2) for corner in perceptron.wrong_corners(gate)]
        lines.append(f"still wrong on {wrong} after {run.passes} passes")
    return lines
