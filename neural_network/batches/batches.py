"""Batches: the examples cut into small groups, one step for each group."""


class Batches:
    """Cuts a list of examples into batches of one size. The last batch holds what is left."""

    def __init__(self, size: int):
        self.size = size

    def cut(self, examples: list) -> list[list]:
        batches = []
        batch = []
        for i in range(len(examples)):
            batch.append(examples[i])
            if len(batch) == self.size:
                batches.append(batch)
                batch = []
        # Examples left over make one last, smaller batch.
        if len(batch) > 0:
            batches.append(batch)
        return batches
