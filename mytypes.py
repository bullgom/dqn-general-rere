from typing import Any
import torch

Size = dict[str, int]
Action = dict[str, int]
ActionSpace = dict[str, int]
Q = dict[str, torch.FloatTensor]
State = Any
Reward = float
Done = bool

SARS = tuple[State, Action, Reward, State, Done]
TupleListSARS = tuple[list[State],
                      list[Action],
                      list[Reward],
                      list[State],
                      list[Done]]
