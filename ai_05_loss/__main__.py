"""The loss functions article as a program: one function per chapter, then the pictures."""

from pathlib import Path

from .article import (
    ANSWER, BEST_LINE, BETS, BIG_MISS, CROSS, FAR_UP_THE_SIDE, LINE_MAP, ON_THE_WALL, POOR_LINE,
    ROOM_HEIGHT, RULE_MAPS, SHARES, SMALL_MISS, SQUARED, SURE_AND_WRONG, SURE_AND_WRONG_RULE,
    THREE_SPOTS, TRUTH,
)
from .examples import EIGHT_EXAMPLES, TEN_POINTS
from .figures import draw_losses, draw_squares, draw_surprise
from .maps import draw_ground, draw_landscape
from .report import (
    both_losses_line, fit_line, gap_lines, share_line, spot_line, surprise_line, two_places,
    worst_misses_line,
)

IMAGES = Path(__file__).parent / "images"


def say(*lines: str, indent: int = 1) -> None:
    for line in lines:
        print("  " * indent + line)


def squared_error() -> None:
    print("Squared error")
    say(*gap_lines(SQUARED, ANSWER, TRUTH))
    small, big = (SQUARED.cost(answer=0, truth=miss) for miss in (SMALL_MISS, BIG_MISS))
    say(f"miss {SMALL_MISS}  costs {two_places(small)}",
        f"miss {BIG_MISS}  costs {two_places(big)}, {big / small:.0f} times as much")
    say(f"ten points, a poor line: {POOR_LINE}")
    say(fit_line(POOR_LINE, TEN_POINTS, SQUARED), worst_misses_line(POOR_LINE, TEN_POINTS, SQUARED), indent=2)
    say(f"the best line: {BEST_LINE}")
    say(fit_line(BEST_LINE, TEN_POINTS, SQUARED), indent=2)


def cross_entropy() -> None:
    print("\nCross entropy")
    say(f"sure and wrong: the truth is yes, the network says {SURE_AND_WRONG}")
    say(both_losses_line((SQUARED, CROSS), SURE_AND_WRONG, TRUTH), indent=2)
    say("surprise, when the truth is yes")
    say(*(surprise_line(bet, mood, CROSS) for bet, mood in BETS.items()), indent=2)
    say("share for the right answer, and its cost")
    say(*(share_line(share, CROSS) for share in SHARES), indent=2)
    halving_step = CROSS.cost(0.5, TRUTH) - CROSS.cost(1, TRUTH)
    say(f"each halving adds {two_places(halving_step)}", indent=2)
    say(f"the network's {two_places(ANSWER)}, the truth is yes")
    say(both_losses_line((SQUARED, CROSS), ANSWER, TRUTH), indent=2)


def the_loss_as_a_landscape() -> None:
    print("\nThe loss as a landscape")
    valley = LINE_MAP.row_at(BEST_LINE.crossing)
    say(f"one number: the crossing held at {two_places(BEST_LINE.crossing)}, only the slope changes")
    say(f"the valley's bottom: slope {two_places(valley.lowest().model.slope)}", indent=2)
    say("two numbers: every spot on the map is one line")
    say(*(spot_line(LINE_MAP.at(line)) for line in THREE_SPOTS), indent=2)
    say(spot_line(LINE_MAP.at(BEST_LINE), ", the lowest"), indent=2)
    say(f"far up the side: {spot_line(LINE_MAP.at(FAR_UP_THE_SIDE))}")
    say(f"cross entropy: every spot is one yes or no rule, on the {len(EIGHT_EXAMPLES)} examples")
    say(f"the lowest: {RULE_MAPS[CROSS].lowest().model}", indent=2)


def main() -> None:
    squared_error()
    cross_entropy()
    the_loss_as_a_landscape()

    draw_squares((SMALL_MISS, BIG_MISS), POOR_LINE, BEST_LINE, TEN_POINTS, SQUARED, IMAGES / "squares.png")
    draw_losses(SQUARED, CROSS, ANSWER, SURE_AND_WRONG, IMAGES / "two_losses.png")
    draw_surprise(SQUARED, CROSS, tuple(BETS), SHARES, IMAGES / "surprise.png")
    draw_landscape(LINE_MAP, THREE_SPOTS + (FAR_UP_THE_SIDE,), BEST_LINE, RULE_MAPS[CROSS], ON_THE_WALL,
                   IMAGES / "landscape.png")
    draw_ground(RULE_MAPS, ROOM_HEIGHT, SURE_AND_WRONG_RULE, IMAGES / "ground.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
