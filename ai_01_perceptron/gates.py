from dataclasses import dataclass

INPUTS_IN_ARTICLE_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class Corner:
    x1: int
    x2: int
    answer: int


@dataclass(frozen=True)
class LogicGate:
    name: str
    corners: tuple[Corner, ...]

    @classmethod
    def from_answers(cls, name: str, answers: tuple[int, int, int, int]) -> "LogicGate":
        corners = tuple(
            Corner(x1, x2, answer)
            for (x1, x2), answer in zip(INPUTS_IN_ARTICLE_ORDER, answers)
        )
        return cls(name, corners)


AND = LogicGate.from_answers("AND", (0, 0, 0, 1))
OR = LogicGate.from_answers("OR", (0, 1, 1, 1))
XOR = LogicGate.from_answers("XOR", (0, 1, 1, 0))
