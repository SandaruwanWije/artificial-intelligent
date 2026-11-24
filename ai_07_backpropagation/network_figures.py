"""The two pictures drawn on the network itself: one number at a time, and the sweep back."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from common.theme import COLORS, panels, save

from .backprop import Sweep
from .drawing import blank_canvas, draw_part, draw_wire, label_bias, label_wire, minus, pretty
from .network import FEEDS, HIDDEN, ForwardPass, Parameters
from .nudge import Nudge
from .report import short, signed

LOSS_MOVES_COLORS = {"comes down": COLORS.accent, "goes up": COLORS.error,
                     "does not move": COLORS.muted}


def draw_number(ax: Axes, name: str, text: str, color: str, width: float = 1.5,
                dotted: bool = False) -> None:
    feed = FEEDS[name]
    if feed.multiplied_by is None:
        label_bias(ax, name, text, color)
        return
    draw_wire(ax, feed.multiplied_by, feed.neuron, color, width, dotted)
    label_wire(ax, name, text, color)


def draw_one_at_a_time(nudges: dict[str, Nudge], parameters: Parameters,
                       passed: ForwardPass, path: Path) -> Path:
    fig, (ax,) = panels(1, width=10, height=6.6)
    blank_canvas(ax)
    for name, nudge in nudges.items():
        text = f"{pretty(name)} {minus(short(parameters.value(name)))}"
        still = nudge.loss_moves == "does not move"
        draw_number(ax, name, text, LOSS_MOVES_COLORS[nudge.loss_moves], dotted=still)
    draw_wire(ax, "y", "loss", COLORS.muted)
    for part in ("x1", "x2", *HIDDEN, "y"):
        draw_part(ax, part, short(passed.value(part)))
    draw_part(ax, "loss", f"{passed.loss:.2f}", ring=COLORS.error)
    handles = [Line2D([], [], color=color, linewidth=2, linestyle=":" if moves == "does not move" else "-")
               for moves, color in LOSS_MOVES_COLORS.items()]
    ax.legend(handles, [f"loss {moves}" for moves in LOSS_MOVES_COLORS], loc="lower right")
    ax.set_title("Turn one number up, hold the other twelve still: which way does the loss move?")
    return save(fig, path)


def draw_sweep(sweep: Sweep, passed: ForwardPass, path: Path) -> Path:
    fig, (ax,) = panels(1, width=10, height=6.6)
    blank_canvas(ax)
    biggest = max(abs(slope) for slope in sweep.slopes.values())
    for name, slope in sweep.slopes.items():
        flat = signed(slope) == "0"
        draw_number(ax, name, f"{pretty(name)} {minus(signed(slope))}",
                    COLORS.muted if flat else COLORS.error,
                    width=1 + 5 * abs(slope) / biggest, dotted=flat)
    draw_wire(ax, "y", "loss", COLORS.error, width=2)
    waiting = {part: minus(signed(number)) for part, number in sweep.waiting_at.items()}
    for part in ("x1", "x2"):
        draw_part(ax, part, short(passed.value(part)))
    for part in (*HIDDEN, "y"):
        draw_part(ax, part, f"{waiting[part]}\nsum {waiting[part + ' sum']}",
                  color=COLORS.error, ring=COLORS.error)
    draw_part(ax, "loss", short(sweep.waiting_at["loss"]), color=COLORS.error, ring=COLORS.error)
    ax.set_title("One sweep back from the loss: a number at every part, a slope on every wire")
    return save(fig, path)
