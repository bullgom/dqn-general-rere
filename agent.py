from abc import ABC, abstractmethod
from selection import Selection
from network import Network
from mytypes import State, Action

class Agent(ABC):
    
    def __init__(
        self, 
        network: Network,
        selection: Selection
    ):
        self.network = network
        self.selection = selection
        
    def step(self, state: State) -> Action:
        q = self.network(state)
        a = self.selection(q)
        return a
