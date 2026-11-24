"""Backpropagation: every slope, first link by link, then all 13 in one sweep back from the loss."""

import math
from dataclasses import dataclass

from .network import FEEDS, HIDDEN, WIRE_INTO_Y, Example, Network


@dataclass(frozen=True)
class Link:
    top: str
    bottom: str
    ratio: float
    why: str


@dataclass(frozen=True)
class Sweep:
    waiting_at: dict[str, float]
    slopes: dict[str, float]


class Backprop:
    def __init__(self, network: Network, example: Example):
        self.parameters = network.parameters
        self.passed = network.forward(example)
        self.right_answer = example.right_answer
        self.links = self.links_from_the_loss_backwards()
        self.link_onward_from = {link.bottom: link for link in self.links}

    def miss(self) -> float:
        return self.passed.y - self.right_answer

    def links_from_the_loss_backwards(self) -> list[Link]:
        y = self.passed.y
        links = [Link("loss", "y", 2 * self.miss(), "2 x the miss"),
                 Link("y", "y sum", y * (1 - y), "sigmoid: y x (1 - y)")]
        for hidden in HIDDEN:
            wire = WIRE_INTO_Y[hidden]
            links.append(Link("y sum", hidden, self.parameters.value(wire), wire))
        for hidden in HIDDEN:
            above_0 = self.passed.value(f"{hidden} sum") > 0
            links.append(Link(hidden, f"{hidden} sum", 1.0 if above_0 else 0.0,
                              "ReLU, above 0" if above_0 else "ReLU, below 0"))
        return links

    def own_link(self, name: str) -> Link:
        feed = FEEDS[name]
        top = f"{feed.neuron} sum"
        if feed.multiplied_by is None:
            return Link(top, name, 1.0, "just added")
        return Link(top, name, self.passed.value(feed.multiplied_by), feed.multiplied_by)

    def road(self, name: str) -> list[Link]:
        road = [self.own_link(name)]
        while road[-1].top != "loss":
            road.append(self.link_onward_from[road[-1].top])
        return road

    def slope_link_by_link(self, name: str) -> float:
        return math.prod(link.ratio for link in self.road(name))

    def sweep(self) -> Sweep:
        waiting_at = {"loss": 1.0}  # the loss moves exactly as much as itself
        for link in self.links:
            waiting_at[link.bottom] = waiting_at[link.top] * link.ratio
        slopes = {}
        for name in self.parameters.names():
            own = self.own_link(name)
            slopes[name] = own.ratio * waiting_at[own.top]
        return Sweep(waiting_at, slopes)
