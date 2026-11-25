import math

from pytest import approx

from neural_network.gradient_descent import GradientDescent
from neural_network.optimizers import Adam, Momentum, Optimizer, RMSProp

START = (-3.2, 0.9)


def ravine_loss(w1: float, w2: float) -> float:
    """Two weights and their loss. Lowest at 0, 0. The walls (w2) are 60 times steeper than the floor (w1)."""
    return 0.5 * (0.1 * w1 * w1 + 6 * w2 * w2)


def far(spot: tuple[float, float]) -> float:
    """How far a spot is from the bottom."""
    return math.sqrt(spot[0] * spot[0] + spot[1] * spot[1])


def walk(optimizer: Optimizer, steps: int = 60) -> list[tuple[float, float]]:
    """Each weight gets a fresh optimizer of its own. Returns every spot, from the start."""
    for_w1, for_w2 = optimizer.fresh(), optimizer.fresh()
    path = [START]
    for _ in range(steps):
        w1, w2 = path[-1]
        path.append((for_w1.step(w1, 0.1 * w1), for_w2.step(w2, 6 * w2)))
    return path


def close_after(path: list[tuple[float, float]]) -> int:
    """The first step after which the walk stays within 0.6 of the bottom."""
    return max(step for step in range(len(path)) if far(path[step]) > 0.6) + 1


def test_plain_steps_zig_zag_between_the_walls_and_crawl_along_the_floor(show):
    path = walk(GradientDescent(learning_rate=0.3))
    show("slopes at the start: w1", 0.1 * -3.2, " w2", 6 * 0.9)
    for step in range(4):
        show("step", step, "  w1", path[step][0], "  w2", path[step][1])
    show("after 60 steps it is still", far(path[60]), "from the bottom")
    assert [round(spot[1], 2) for spot in path[:4]] == [0.9, -0.72, 0.58, -0.46]
    assert [round(spot[0], 2) for spot in path[:4]] == [-3.2, -3.1, -3.01, -2.92]
    assert far(path[60]) == approx(0.51, abs=0.005)


def test_a_learning_rate_past_2_over_6_climbs_out_of_the_ravine(show):
    path = walk(GradientDescent(learning_rate=0.34), steps=3)
    losses = [ravine_loss(w1, w2) for w1, w2 in path]
    show("each step multiplies w2 by 1 - 6 x 0.34 =", 1 - 6 * 0.34)
    show("w2", [spot[1] for spot in path], "  loss", losses)
    assert [round(loss, 2) for loss in losses[:3]] == [2.94, 3.11, 3.29]


def test_momentum_steps_that_agree_grow_and_a_step_backward_almost_cancels(show):
    momentum, weight, steps = Momentum(learning_rate=1.0), 0.0, []
    for _ in range(4):
        moved = momentum.step(weight, slope=-1.0)  # a plain step of +1 every time
        steps.append(moved - weight)
        weight = moved
    show("plain steps of +1 give steps of", steps, " growing toward 1 / (1 - 0.9) = 10")
    assert steps == approx([1.0, 1.9, 2.71, 3.439])
    turned = Momentum(learning_rate=1.0)
    turned.velocity = 1.0
    moved = turned.step(0.0, slope=1.0)  # now the plain step is -1
    show("last step 1.0, keep 90% = 0.9, add a plain step of -1 -> new step", moved)
    assert moved == approx(-0.1)


def test_momentum_speeds_up_along_the_floor_and_is_slow_to_turn_on_the_walls(show):
    path = walk(Momentum(learning_rate=0.1))
    w1_steps = [path[step + 1][0] - path[step][0] for step in range(10)]
    show("steps along w1:", w1_steps)
    assert (round(w1_steps[0], 2), round(w1_steps[9], 2)) == (0.03, 0.17)
    plain, kept = -0.1 * 6 * path[2][1], 0.9 * (path[2][1] - path[1][1])
    show("step 3 for w2: plain step", plain, "  kept", kept, "  new step", path[3][1] - path[2][1])
    assert (round(plain, 2), round(kept, 2), round(path[3][1] - path[2][1], 2)) == (0.21, -0.63, -0.43)
    show("it rolls past the bottom to w1 =", max(spot[0] for spot in path), "  after 60 steps:", far(path[60]))
    assert max(spot[0] for spot in path) == approx(0.49, abs=0.005)
    assert far(path[60]) == approx(0.03, abs=0.005)


def test_rmsprop_gives_each_weight_a_learning_rate_of_its_own(show):
    rmsprop, weight = RMSProp(learning_rate=0.1), 0.9
    moved = rmsprop.step(weight, slope=5.4)
    show("first slope 5.4: running average of squares", rmsprop.average_square,
         "  its root", math.sqrt(rmsprop.average_square), "  step", moved - weight)
    assert rmsprop.average_square == approx(2.916)
    assert moved - weight == approx(-0.3162, abs=0.0001)  # bigger than 0.1: the memory started empty
    for step in range(1, 60):
        slope = 5.4 if step % 2 == 0 else -5.4  # the same size, on the other wall each time
        moved = rmsprop.step(weight, slope)
        step_taken = moved - weight
    usual_size = math.sqrt(rmsprop.average_square)
    show("after 60 slopes of 5.4, -5.4, ...: usual size", usual_size, "  own learning rate", 0.1 / usual_size,
         "  step", step_taken)
    assert usual_size == approx(5.4, abs=0.01)
    assert abs(step_taken) == approx(0.1, abs=0.001)
    path = walk(RMSProp(learning_rate=0.1))
    show("on the ravine, first steps: w1", path[1][0] - path[0][0], " w2", path[1][1] - path[0][1])
    assert path[1][0] - path[0][0] == approx(0.3162, abs=0.0001)
    assert far(path[40]) < 0.1


def test_adams_first_step_is_the_learning_rate_for_both_weights(show):
    adam = Adam(learning_rate=0.1)
    moved = adam.step(0.9, slope=5.4)
    show("w2: average slope", adam.average_slope, " / how full", 1 - adam.slope_empty, "=",
         adam.average_slope / (1 - adam.slope_empty), "  step", moved - 0.9)
    assert adam.average_slope == approx(0.54)
    assert adam.average_slope / (1 - adam.slope_empty) == approx(5.4)
    assert moved - 0.9 == approx(-0.1)
    path = walk(Adam(learning_rate=0.1), steps=3)
    show("on the ravine: w1", [spot[0] for spot in path], "  w2", [spot[1] for spot in path])
    assert [round(spot[0], 2) for spot in path] == [-3.2, -3.1, -3.0, -2.9]
    assert [round(spot[1], 2) for spot in path] == [0.9, 0.8, 0.7, 0.6]


def test_adam_steps_fully_when_slopes_agree_and_barely_when_they_cancel(show):
    agree, cancel, weight = Adam(learning_rate=0.1), Adam(learning_rate=0.1), 0.0
    for slope in (5, 5, 5):
        step_when_agreeing = agree.step(weight, slope) - weight
    for slope in (5, -5, 5, -5):
        step_when_cancelling = cancel.step(weight, slope) - weight
    average = cancel.average_slope / (1 - cancel.slope_empty)
    show("slopes 5, 5, 5:      step", step_when_agreeing)
    show("slopes 5, -5, 5, -5: average slope", average, "  step", step_when_cancelling)
    assert step_when_agreeing == approx(-0.1)
    assert average == approx(-0.26, abs=0.005)
    assert step_when_cancelling == approx(0.0053, abs=0.0001)


def test_all_four_on_the_ravine_for_60_steps(show):
    results = []
    for optimizer in (GradientDescent(0.3), Momentum(0.1), RMSProp(0.1), Adam(0.1)):
        path = walk(optimizer)
        show(type(optimizer).__name__, " close after", close_after(path), "steps   distance after 60:", far(path[60]))
        results.append((close_after(path), round(far(path[60]), 2)))
    assert results == [(55, 0.51), (20, 0.03), (25, 0.0), (30, 0.16)]


def test_a_fresh_optimizer_keeps_the_settings_and_starts_with_an_empty_memory(show):
    used = Momentum(learning_rate=0.1, keep=0.8)
    used.step(1.0, slope=2.0)
    fresh = used.fresh()
    show("used: velocity", used.velocity, "  fresh: velocity", fresh.velocity, " keep", fresh.keep)
    assert (used.velocity, fresh.velocity) == (approx(-0.2), 0.0)
    assert (fresh.learning_rate, fresh.keep) == (0.1, 0.8)
