from abc import ABC, abstractmethod
from typing import Any, Callable, Generator
import numpy as np
import torch
from mytypes import Action


class Selection(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, q: dict[str, torch.FloatTensor]) -> dict[str, Action]:
        raise NotImplementedError


class EpsilonGenerator(ABC):

    @abstractmethod
    def step(self) -> float:
        raise NotImplementedError


class LinearEpsilonGenerator(EpsilonGenerator):

    def __init__(self, start: float, end: float, steps: int) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        self.decrease_per_step = (end-start)/steps
        self.current = start

    def step(self) -> float:
        self.current -= self.decrease_per_step
        return self.current


class EpsilonGreedySelection(Selection):

    def __init__(
        self,
        e_generator: EpsilonGenerator,
        action_space_sizes: dict[str, int]
    ):
        super().__init__()
        self.e_generator = e_generator
        self.action_space_sizes = action_space_sizes

    def __call__(self, q: dict[str, torch.FloatTensor]) -> dict[str, Action]:
        e = self.e_generator.step()
        samples = np.random.uniform(low=0, high=1, size=(len(q),))
        selected_actions = {}
        for key, prob in zip(q.keys(), samples):

            if prob < e:
                a = np.random.randint(0, len(q[key]))
            else:
                a = np.argmax(q)
            selected_actions[key] = a
        return selected_actions


if __name__ == "__main__":

    egen = LinearEpsilonGenerator(0.9, 0.1, 333)
    selection = EpsilonGreedySelection(egen, {"A": 2, "B": 4})

    for i in range(10):
        actions = selection({"A": np.random.uniform(size=(2,)),
                            "B": np.random.uniform(size=4)})
        print(actions)