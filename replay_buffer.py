from abc import ABC, abstractmethod
from mytypes import SARS, ConcatSARS, ActionSpace
import numpy as np
import random
import torch


class ReplayBuffer(ABC):

    def __init__(
        self, 
        capacity: int, 
        action_space: ActionSpace, 
        saving_device: torch.device,
        using_device: torch.device
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.capacity = capacity
        self.saving_device = saving_device
        self.using_device = using_device
        self.buffer: list[SARS] = []

    def append(self, sars: SARS):
        if self.full():
            self.buffer.pop(0)
        sars = self.detach_sars(sars)
        self.buffer.append(sars)
        
    def detach_sars(self, sars: SARS) -> SARS:
        s, a, r, sn, d = sars
        s = s.to(self.saving_device).detach()
        sn = sn.to(self.saving_device).detach()
        return s, a, r, sn, d

    def sample(self, batch_size: int) -> ConcatSARS:
        if batch_size > self.capacity:
            raise ValueError(f"You can't request more than we have sir")
        samples = random.choices(self.buffer, k=batch_size)
        s, a, r, sn, d = tuple(zip(*samples))

        s = torch.concat(s).to(self.using_device)
        r = torch.tensor(r).to(self.using_device)
        sn = torch.concat(sn).to(self.using_device)
        d = torch.tensor(d, dtype=torch.bool).to(self.using_device)
        # I know. Yuk!
        a = {k: torch.tensor([row[k] for row in a], dtype=torch.int64).to(self.using_device)
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
