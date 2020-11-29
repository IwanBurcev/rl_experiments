import random
from collections import deque
import numpy as np


class PrioretizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_sate, done):
        self.buffer.append((state, action, reward, next_sate, done))
        self.priorities.append(max(self.priorities, default=1))

    def get_probabililties(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilites):
        importance = 1 / len(self.buffer) * 1 / probabilites
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=0.1):
        sample_size = min(len(self), batch_size)
        sample_probs = self.get_probabililties(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.asarray(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(np.asarray, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e.item()) + offset


    def __len__(self):
        return len(self.buffer)
