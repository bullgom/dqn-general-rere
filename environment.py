from preprocessing import Preprocessing
from abc import ABC, abstractmethod
from typing import Any, Optional
import gym

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

class CartPole(Environment):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_ = gym.make("CartPole-v1")
        
    def step(self, action: int):
        _, r, done, _ = self.env_.step(action)
        self.env_.render("")
        
        
    
        
    