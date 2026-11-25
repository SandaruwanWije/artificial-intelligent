"""Split: the examples in three piles, so a network is graded on examples it never trained on."""

from neural_network.batches import Shuffle


class Split:
    """Shuffles the examples, then cuts them into a training, a validation and a test pile."""

    def __init__(self, examples: list, validation_size: int, test_size: int, shuffle: Shuffle):
        # Shuffle first, so each pile is a fair mix of all the examples.
        mixed = shuffle.shuffled(examples)
        training_size = len(mixed) - validation_size - test_size
        self.training = []  # what the network learns from
        self.validation = []  # what the settings are chosen by
        self.test = []  # kept away until every choice is made, then used once
        for i in range(len(mixed)):
            if i < training_size:
                self.training.append(mixed[i])
            elif i < training_size + validation_size:
                self.validation.append(mixed[i])
            else:
                self.test.append(mixed[i])
