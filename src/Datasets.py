import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ExperienceBuffer:
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def append(self, transitions):
        for transition in transitions:
            if len(self.buffer) == self.capacity:
                self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size, remove=False, shuffle=False):
        if shuffle:
            random.shuffle(self.buffer)
        if remove:
            samples = self.buffer[:batch_size]
            self.remove(samples)
        else:
            samples = self.buffer[-batch_size:]
        return samples

    def remove(self, samples):
        for sample in samples:
            self.buffer.remove(sample)

    def __len__(self):
        return len(self.buffer)
