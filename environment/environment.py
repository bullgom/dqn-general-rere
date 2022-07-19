from preprocessing import Preprocessing
from abc import ABC, abstractmethod
from typing import Any, Optional

State = Any

class Environment(ABC):
    
    def __init__(self, preps: list[Preprocessing] = None) -> None:
        # Preprocessings 
        self.preps = preps if preps else []
    
    @abstractmethod
    def step(self, action: int) -> tuple[State, float, bool]:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_original_state_size(self) -> tuple[int]:
        raise NotImplementedError

    def state_size(self) -> tuple[int]:
        size = self.get_original_state_size()
        for prep in self.preps:
            size = prep.output_size(size)
        return size
