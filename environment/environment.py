from preprocessing import Preprocessing
from abc import ABC, abstractmethod
from typing import Any, Optional

State = Any

class Environment(ABC):
    
    def __init__(self, preps: list[Preprocessing] = None) -> None:
        # Preprocessings 
        self.preps = preps if preps else []
    
    @abstractmethod
    def step(self, action: int) -> State:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

        
    