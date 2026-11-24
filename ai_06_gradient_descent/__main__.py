"""The gradient descent article as a program, one function per chapter."""

from pathlib import Path

from .bowl import Bowl, Weights
from .bowl_figures import draw_gradient, draw_learning_rates, draw_rate_losses
from .descent import BowlDescent, GradientDescent, ValleyDescent
from .feel import feel_gradient, feel_slope
from .report import (climb_line, gradient_lines, loss_table, slope_line,
                     step_against_line, step_line, step_sizes_line, uphill_line,
                     weight_line, weights_line)
from .valley import Valley
from .valley_figures import draw_slopes, draw_steps

IMAGES = Path(__file__).parent / "images"

VALLEY = Valley()
VALLEY_START = -2.2
VALLEY_DESCENT = GradientDescent(learning_rate=0.8)

BOWL = Bowl(steepness_across=6.0)
BOWL_START = Weights(w1=-3.2, w2=0.9)
BOWL_DESCENT = GradientDescent(learning_rate=0.15)
LEARNING_RATES = {"too small": 0.03, "about right": 0.15, "too big": 0.34}
STEPS_ON_THE_CHART = 25


def downhill_in_the_fog() -> float:
    print("Downhill in the fog")
    print(f"  {weight_line(VALLEY_START, VALLEY.loss(VALLEY_START))}")
    one_step_on = VALLEY_DESCENT.step(VALLEY_START, VALLEY.slope(VALLEY_START))
    print(f"  one step downhill: weight {one_step_on:.2f}")
    return one_step_on


def the_slope(spots: tuple[float, ...]) -> None:
    print("\nThe slope")
    for weight in spots:
        felt = feel_slope(VALLEY.loss, weight)
        print(f"  {slope_line(weight, VALLEY.slope(weight), felt)}")


def one_step_at_a_time() -> ValleyDescent:
    print("\nOne step at a time")
    descent = VALLEY_DESCENT.down_the_valley(VALLEY, VALLEY_START, steps=5)
    print(f"  {uphill_line(descent.learning_rate, descent.steps[1])}")
    for step in descent.steps[:4]:
        print(f"  {step_line(step)}")
    print(f"  {step_sizes_line(descent)}")
    return descent


def two_weights_the_gradient() -> None:
    print("\nTwo weights: the gradient")
    print(f"  start: {weights_line(BOWL_START, BOWL.loss(BOWL_START))}")
    for line in gradient_lines(BOWL.gradient(BOWL_START), feel_gradient(BOWL, BOWL_START)):
        print(f"  {line}")
    after = BOWL_DESCENT.across_the_bowl(BOWL, BOWL_START, steps=1).path[1]
    print(f"  {step_against_line(BOWL_DESCENT.learning_rate, BOWL_START, after)}")
    print(f"  after one step: {weights_line(after, BOWL.loss(after))}")


def picking_the_step_size() -> dict[str, BowlDescent]:
    print("\nPicking the step size")
    descents = {
        name: GradientDescent(rate).across_the_bowl(BOWL, BOWL_START, STEPS_ON_THE_CHART)
        for name, rate in LEARNING_RATES.items()
    }
    for line in loss_table(descents, every=5):
        print(f"  {line}")
    print(f"  {climb_line('too big', descents['too big'])}")
    return descents


def main() -> None:
    one_step_on = downhill_in_the_fog()
    the_slope((VALLEY_START, one_step_on))
    valley_descent = one_step_at_a_time()
    two_weights_the_gradient()
    bowl_descents = picking_the_step_size()

    draw_slopes(VALLEY, (VALLEY_START, one_step_on), IMAGES / "slope.png")
    draw_steps(VALLEY, valley_descent, IMAGES / "valley_steps.png")
    draw_gradient(BOWL, BOWL_START, BOWL_DESCENT.learning_rate, IMAGES / "gradient.png")
    draw_learning_rates(BOWL, bowl_descents, IMAGES / "learning_rates.png")
    draw_rate_losses(bowl_descents, IMAGES / "rate_losses.png")
    print(f"\nPictures saved to {IMAGES}")


if __name__ == "__main__":
    main()
