from abc import ABC, abstractmethod
from mytypes import SARS, TupleListSARS
import numpy as np
import random

class ReplayBuffer(ABC):
    
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.buffer : list[SARS] = []
    
    def append(self, sars: SARS):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(sars)
    
    def sample(self, batch_size: int) -> TupleListSARS:
        if batch_size > self.capacity:
            raise ValueError(f"You can't request more than we have sir")
        samples = random.choices(self.buffer, k=batch_size)
        tl_sars = tuple(zip(*samples))
        return tl_sars        
        
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
