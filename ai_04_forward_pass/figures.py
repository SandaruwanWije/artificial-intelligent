"""The charts for the article, in the blog's colours (see common/theme.py)."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.patches import Patch

from common.theme import COLORS, panels, quiet, save

from .answer import Gap
from .network import ForwardPass
from .neuron import Working
from .work import Chip, Step

SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def pretty(name: str) -> str:
    return name.translate(SUBSCRIPTS)


def shown(value: float) -> str:
    return f"{round(value, 3):g}".replace("-", "−")


def write_at_end(ax: Axes, at: float, end: float, label: str, above: bool, **style: object) -> None:
    ax.text(at, end + (0.04 if above else -0.04), label, ha="center",
            va="bottom" if above else "top", **style)


def draw_waterfall(ax: Axes, working: Working) -> None:
    neuron = working.neuron
    steps = [(f"{shown(w)}\n×{shown(x)}", p)
             for w, x, p in zip(neuron.weights, working.inputs, working.products)]
    steps.append(("bias", neuron.bias))
    running = 0.0
    for at, (_, step) in enumerate(steps):
        ax.bar(at, step, bottom=running, width=0.6, color=COLORS.class_0, alpha=0.45)
        write_at_end(ax, at, running + step, shown(step), step >= 0, fontsize=9, color=COLORS.muted)
        running += step
    z_at, value_at = len(steps), len(steps) + 1
    ax.bar(z_at, working.z, width=0.6, color=COLORS.class_0)
    write_at_end(ax, z_at, working.z, shown(working.z), working.z >= 0, fontsize=10, fontweight="bold")
    ax.bar(value_at, working.value, width=0.6, color=COLORS.accent)
    write_at_end(ax, value_at, working.value, shown(round(working.value, 2)), True,
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color=COLORS.muted, linewidth=1)
    names = [name for name, _ in steps] + ["z", neuron.activation.name]
    ax.set_xticks(range(len(names)), names, fontsize=9)
    ax.set_ylim(-0.8, 1.05)
    off = ", switched off" if working.switched_off else ""
    ax.set_title(f"{pretty(neuron.name)} → {shown(round(working.value, 2))}{off}",
                 color=COLORS.muted if working.switched_off else COLORS.ink)
    quiet(ax)


def draw_sums(run: ForwardPass, path: Path) -> Path:
    fig, axes = panels(len(run.workings), width=3.6, height=3.8)
    for ax, working in zip(axes, run.workings):
        draw_waterfall(ax, working)
    axes[0].set_ylabel("running total")
    return save(fig, path)


def draw_work(schedules: dict[Chip, list[Step]], path: Path) -> Path:
    longest = max(len(steps) for steps in schedules.values())
    tallest = max(step.multiplications for steps in schedules.values() for step in steps)
    layer_colors = (COLORS.class_0, COLORS.accent)
    fig, axes = panels(len(schedules), width=4.6, height=3.6)
    for ax, (chip, steps) in zip(axes, schedules.items()):
        for number, step in enumerate(steps, start=1):
            ax.bar(number, step.multiplications, width=0.6, color=layer_colors[step.layer])
            ax.text(number, step.multiplications + 0.12, str(step.multiplications),
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{chip.name}: {len(steps)} steps")
        ax.set(xlim=(0.4, longest + 0.6), ylim=(0, tallest + 1), xlabel="step",
               xticks=range(1, longest + 1), yticks=range(0, tallest + 1, 2))
        quiet(ax)
    axes[0].set_ylabel("multiplications in the step")
    layers = [Patch(color=c, label=name) for c, name in zip(layer_colors, ("hidden layer", "output layer"))]
    axes[-1].legend(handles=layers, loc="upper right")
    return save(fig, path)


def draw_gap(gap: Gap, path: Path) -> Path:
    fig, (ax,) = panels(1, width=8.0, height=2.6)
    ax.plot([0, 1], [0, 0], color=COLORS.muted, linewidth=2)
    ax.plot([gap.guess, gap.right_answer], [0, 0], color=COLORS.error, linewidth=5)
    ax.scatter([gap.guess], [0], s=160, color=COLORS.accent, zorder=3)
    ax.scatter([gap.right_answer], [0], s=160, color=COLORS.class_0, zorder=3)
    ax.text(gap.guess, -0.35, f"the network: {gap.guess:.2f}", ha="center", va="top")
    ax.text(gap.right_answer, -0.35, f"right answer: {shown(gap.right_answer)}", ha="center", va="top")
    middle = (gap.guess + gap.right_answer) / 2
    ax.text(middle, 0.3, f"{gap.short_by:.2f} short", ha="center", va="bottom",
            color=COLORS.error, fontweight="bold")
    ax.set_title("How wrong is it? The answer against the right one")
    ax.set(xlim=(-0.05, 1.12), ylim=(-1, 1), yticks=[])
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    return save(fig, path)
