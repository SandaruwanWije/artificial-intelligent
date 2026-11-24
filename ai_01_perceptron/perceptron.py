"""The perceptron: one neuron with a hard edge (Rosenblatt, 1958).

It takes two signals, x1 and x2, each 0 or 1. Each signal is multiplied by
its weight, the results are added up, and the bias is added on top. That
total is called z. If z is above zero the answer is 1, otherwise it is 0.
"""


class Perceptron:
    def __init__(self, w1, w2, bias, rate=0.5):
        self.w1 = w1      # how much signal x1 counts
        self.w2 = w2      # how much signal x2 counts
        self.bias = bias  # a head start, which can be below zero
        self.rate = rate  # the learning rate: how big each nudge is

    def total(self, x1, x2):
        """z: each signal times its weight, added up, plus the bias."""
        return self.w1 * x1 + self.w2 * x2 + self.bias

    def guess(self, x1, x2):
        """The hard edge: 1 if z is above zero, otherwise 0."""
        z = self.total(x1, x2)
        return 1 if z > 0 else 0

    def train(self, examples, max_passes=8, show_fixes=False):
        """Rosenblatt's learning rule, one example at a time.

        One pass checks every example once. It keeps making passes until a
        whole pass goes by with no mistakes, or gives up after max_passes.
        Returns how many mistakes each pass made.
        """
        mistakes_per_pass = []
        for pass_number in range(1, max_passes + 1):
            mistakes = 0
            for x1, x2, answer in examples:
                guess = self.guess(x1, x2)
                error = answer - guess  # 1, 0 or -1

                # weight <- weight + rate x error x its signal
                # bias   <- bias + rate x error
                # A right answer has an error of 0, so nothing moves.
                self.w1 = self.w1 + self.rate * error * x1
                self.w2 = self.w2 + self.rate * error * x2
                self.bias = self.bias + self.rate * error

                if error != 0:
                    mistakes += 1
                    if show_fixes:
                        print(f"  pass {pass_number}  ({x1}, {x2})  guessed {guess}, "
                              f"wants {answer}  ->  {self}")
            mistakes_per_pass.append(mistakes)
            if mistakes == 0:
                break
        return mistakes_per_pass

    def __str__(self):
        return f"w1 {self.w1:g}, w2 {self.w2:g}, bias {self.bias:g}"
