"""A layer with no bend: it multiplies x by a weight and adds a bias."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    w: float
    b: float

    def __call__(self, x: float) -> float:
        return self.w * x + self.b

    def then(self, next_layer: "Layer") -> "Layer":
        return Layer(w=next_layer.w * self.w, b=next_layer.w * self.b + next_layer.b)

    def __str__(self) -> str:
        return f"w {self.w:g}, b {self.b:g}"


def stack(layers: list[Layer]) -> Layer:
    together = layers[0]
    for layer in layers[1:]:
        together = together.then(layer)
    return together
