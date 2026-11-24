"""The article's numbers, in one place."""

import math

from .examples import EIGHT_EXAMPLES, TEN_POINTS
from .landscape import Landscape, spaced
from .losses import CrossEntropy, SquaredError
from .models import Line, Rule

ARTICLE_7_LAST_Z = 0.355
ANSWER = 1 / (1 + math.exp(-ARTICLE_7_LAST_Z))
TRUTH = 1
SMALL_MISS, BIG_MISS = 0.3, 0.6
SURE_AND_WRONG = 0.01
BETS = {0.99: "hardly surprised", 0.59: "a little surprised", 0.1: "quite surprised", 0.01: "shocked"}
SHARES = (1, 0.5, 0.25, 0.125)

SQUARED = SquaredError()
CROSS = CrossEntropy()

POOR_LINE = Line(slope=0.4, crossing=1.6)
BEST_LINE = Line.best_fit(TEN_POINTS)
THREE_SPOTS = (Line(0.3, 2.0), Line(1.1, 0.9), Line(1.6, -0.9))
FAR_UP_THE_SIDE = Line(1.7, 2.0)
LINE_MAP = Landscape(SQUARED, Line, TEN_POINTS, across=spaced(-0.2, 1.9), up=spaced(-1.6, 2.3))

WEIGHTS, BIASES = spaced(-3.5, 6), spaced(-3.5, 3.2)
RULE_MAPS = {loss: Landscape(loss, Rule, EIGHT_EXAMPLES, WEIGHTS, BIASES) for loss in (SQUARED, CROSS)}
ON_THE_WALL = Rule(weight=-3, bias=1)
SURE_AND_WRONG_RULE = Rule(weight=-3, bias=1.5)
ROOM_HEIGHT = 4
