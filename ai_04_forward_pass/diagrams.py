"""The network and the matrices, drawn as diagrams with every number in its place."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

from common.theme import COLORS, panels, save

from .figures import pretty, shown
from .matrix import MatrixLayer, MatrixNetwork
from .network import ForwardPass, Network

RADIUS = 0.42
CELL = 0.8
COLUMN_X = (0.0, 4.0, 8.0)
ROWS_Y = {1: (1.7,), 2: (2.8, 0.6), 3: (3.4, 1.7, 0.0)}


def draw_node(ax: Axes, at: tuple[float, float], value: str, name: str, under: str, edge: str) -> None:
    x, y = at
    ax.add_patch(Circle(at, RADIUS, facecolor=COLORS.background, edgecolor=edge, linewidth=2, zorder=3))
    ax.text(x, y, value, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    ax.text(x, y + RADIUS + 0.12, name, ha="center", va="bottom", fontsize=11, color=edge)
    ax.text(x, y - RADIUS - 0.12, under, ha="center", va="top", fontsize=9, color=COLORS.muted)


def node_color(switched_off: bool, is_output: bool) -> str:
    if is_output:
        return COLORS.accent
    return COLORS.muted if switched_off else COLORS.class_0


def draw_network(run: ForwardPass, path: Path) -> Path:
    fig, (ax,) = panels(1, width=9.5, height=5.4)
    inputs = [(COLUMN_X[0], y) for y in ROWS_Y[len(run.inputs)]]
    places = [inputs] + [[(COLUMN_X[i + 1], y) for y in ROWS_Y[len(layer)]]
                         for i, layer in enumerate(run.layers)]
    for i, (name, meaning) in enumerate((("x₁", "how yellow"), ("x₂", "how soft"))):
        draw_node(ax, inputs[i], shown(run.inputs[i]), name, meaning, COLORS.muted)
    for layer, (froms, tos) in enumerate(zip(places, places[1:])):
        senders = run.inputs if layer == 0 else run.values(layer - 1)
        for working, to in zip(run.layers[layer], tos):
            for weight, start, sent in zip(working.neuron.weights, froms, senders):
                quiet_wire = sent == 0
                ax.plot(*zip(start, to), color=COLORS.grid if quiet_wire else COLORS.muted,
                        linestyle="--" if quiet_wire else "-", linewidth=1.3, zorder=1)
                label_at = np.array(start) + 0.72 * (np.array(to) - np.array(start))
                ax.text(*label_at, shown(weight), ha="center", va="center", fontsize=10, zorder=2,
                        bbox=dict(facecolor=COLORS.background, edgecolor="none", pad=1.5))
            edge = node_color(working.switched_off, is_output=layer == len(run.layers) - 1)
            under = f"b {shown(working.neuron.bias)} · z {shown(working.z)}"
            draw_node(ax, to, shown(round(working.value, 2)), pretty(working.neuron.name), under, edge)
    ax.text(COLUMN_X[2], places[-1][0][1] + 1.15, f"ripe? {run.answer:.0%} sure", ha="center",
            fontsize=10, color=COLORS.accent)
    x1, x2 = run.inputs
    ax.set_title(f"One forward pass: {shown(x1)} and {shown(x2)} in, {run.answer:.2f} out")
    ax.set(xlim=(-0.9, 8.9), ylim=(-0.9, 4.1), aspect="equal")
    ax.axis("off")
    return save(fig, path)


def draw_grid(ax: Axes, left: float, middle: float, cells: np.ndarray, caption: str,
              lit_row: int | None = None, dim_rows: tuple[int, ...] = ()) -> float:
    rows, columns = cells.shape
    top = middle + rows * CELL / 2
    for r in range(rows):
        lit = r == lit_row
        text_color = COLORS.accent if lit else COLORS.muted if r in dim_rows else COLORS.ink
        for c in range(columns):
            ax.add_patch(Rectangle((left + c, top - (r + 1) * CELL), 1, CELL, facecolor="none",
                                   edgecolor=COLORS.accent if lit else COLORS.grid))
            ax.text(left + c + 0.5, top - (r + 0.5) * CELL, shown(cells[r, c]),
                    ha="center", va="center", fontsize=11, color=text_color)
    ax.text(left + columns / 2, top - rows * CELL - 0.2, caption, ha="center", va="top",
            fontsize=10, color=COLORS.muted)
    return left + columns


def draw_layer(ax: Axes, middle: float, names: list[str], layer: MatrixLayer,
               x: np.ndarray, out: np.ndarray, symbols: tuple[str, str, str, str]) -> None:
    w_name, x_name, b_name, out_name = symbols
    for r, name in enumerate(names):
        ax.text(-0.25, middle + (len(names) / 2 - r - 0.5) * CELL, pretty(name),
                ha="right", va="center", color=COLORS.muted)
    switched_off = tuple(r for r, value in enumerate(out) if value == 0)
    rows, columns = layer.W.shape
    pieces = [
        (layer.W, f"{w_name} {rows} × {columns}", "·"),
        (x[:, None], x_name, "+"),
        (layer.b[:, None], b_name, "="),
        (layer.z(x)[:, None], "z", f"{layer.activation.name}\n→"),
        (np.round(out, 2)[:, None], out_name, ""),
    ]
    left = 0.0
    for cells, caption, operator in pieces:
        one_per_neuron = cells.shape[0] == rows
        right = draw_grid(ax, left, middle, cells, caption, 0 if one_per_neuron else None,
                          switched_off if one_per_neuron else ())
        ax.text(right + 0.55, middle, operator, ha="center", va="center",
                fontsize=18 if len(operator) == 1 else 10)
        left = right + 1.1


def draw_matrix(network: Network, matrix: MatrixNetwork, by_matrix: list[np.ndarray],
                inputs: tuple[float, ...], path: Path) -> Path:
    fig, (ax,) = panels(1, width=9.6, height=6.2)
    layer_inputs = [np.array(inputs), *by_matrix[:-1]]
    symbols = (("W₁", "x", "b₁", "h"), ("W₂", "h", "b₂", "y"))
    titles = ("hidden layer:  h = ReLU(W₁ · x + b₁)", "output layer:  y = sigmoid(W₂ · h + b₂)")
    rows = zip((5.0, 0.6), network.layers, matrix.layers, layer_inputs, by_matrix, symbols, titles)
    for middle, layer, matrix_layer, x, out, layer_symbols, title in rows:
        names = [neuron.name for neuron in layer.neurons]
        draw_layer(ax, middle, names, matrix_layer, x, out, layer_symbols)
        ax.text(-1.2, middle + 1.75, title, fontsize=12, fontweight="bold")
    ax.set_title("A layer in one move: each row of W is one neuron")
    ax.set(xlim=(-1.3, 12.8), ylim=(-0.75, 7.4), aspect="equal")
    ax.axis("off")
    return save(fig, path)
