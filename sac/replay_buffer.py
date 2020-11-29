import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_sate, done):
        self.buffer.append((state, action, reward, next_sate, done))

    def sample(self, batch_size):
        sample_size = min(len(self), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.buffer)

