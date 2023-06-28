import random
from collections import namedtuple

Transition = tuple

class ExperienceBuffer:
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.buffer = []

    def append(self, transitions: list[Transition]):
        for transition in transitions:
            if len(self.buffer) == self.capacity:
                self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size: int, remove: bool=False, shuffle: bool=False) -> list[Transition]:
        if shuffle:
            random.shuffle(self.buffer)
        if remove:
            samples = self.buffer[:batch_size]
            self.remove(samples)
        else:
            samples = self.buffer[-batch_size:]
        return samples

    def remove(self, samples: list[Transition]):
        for sample in samples:
            self.buffer.remove(sample)

    def empty(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
