from abc import ABC, abstractmethod
from mytypes import SARS, ConcatSARS, ActionSpace
import numpy as np
import random
import torch


class ReplayBuffer(ABC):

    def __init__(self, capacity: int, action_space: ActionSpace) -> None:
        super().__init__()
        self.action_space = action_space
        self.capacity = capacity
        self.buffer: list[SARS] = []

    def append(self, sars: SARS):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(sars)

    def sample(self, batch_size: int) -> ConcatSARS:
        if batch_size > self.capacity:
            raise ValueError(f"You can't request more than we have sir")
        samples = random.choices(self.buffer, k=batch_size)
        s, a, r, sn, d = tuple(zip(*samples))

        s = torch.concat(s)
        r = torch.tensor(r)
        sn = torch.concat(sn)
        d = torch.tensor(d, dtype=torch.bool)
        a = {k: torch.tensor([row[k] for row in a], dtype=torch.int64) 
                              for k in self.action_space.keys()}

        return s, a, r, sn, d

    def full(self) -> bool:
        return len(self.buffer) == self.capacity

    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


if __name__ == "__main__":
    buffer = ReplayBuffer(10)
    for i in range(20):
        buffer.append((f"{i},0", f"{i}, 1", f"{i}, 2", f"{i}, 3"))
    s = buffer.sample(3)
    a = 1
    pass
