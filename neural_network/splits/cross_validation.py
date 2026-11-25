"""Cross validation: every example is used for training in some runs and for validation in one."""


class CrossValidation:
    """Cuts the examples into folds. In each run one fold is the validation set and the rest is the training set."""

    def __init__(self, examples: list, folds: int):
        self.folds = []
        start = 0
        for f in range(folds):
            # The folds still to make share the examples still left, so no fold is more than 1 bigger than another.
            folds_left = folds - f
            size = (len(examples) - start) // folds_left
            fold = []
            for i in range(start, start + size):
                fold.append(examples[i])
            self.folds.append(fold)
            start = start + size

    def validation(self, run: int) -> list:
        """The fold whose turn it is in this run."""
        return self.folds[run]

    def training(self, run: int) -> list:
        """Every other fold, put together."""
        training = []
        for f in range(len(self.folds)):
            if f != run:
                for i in range(len(self.folds[f])):
                    training.append(self.folds[f][i])
        return training
