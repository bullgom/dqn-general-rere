from preprocessing import Preprocessing
from abc import ABC, abstractmethod
from typing import Any, Optional
from mytypes import Size, State, ActionSpace, Action


class Environment(ABC):

    def __init__(self, preps: list[Preprocessing] = None) -> None:
        # Preprocessings
        self.preps = preps if preps else []

    @abstractmethod
    def step(self, action: Action) -> tuple[State, float, bool]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def original_state_size(self) -> Size:
        raise NotImplementedError

    def state_size(self) -> Size:
        size = self.original_state_size()
        for prep in self.preps:
            size = prep.output_size(size)
        return size

    @abstractmethod
    def action_space(self) -> ActionSpace:
        raise NotImplementedError
