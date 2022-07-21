from typing import Any
import torch

Size = dict[str, int]
Action = dict[str, int]
ActionSpace = dict[str, int]
Q = dict[str, torch.FloatTensor]
State = Any
