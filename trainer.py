from abc import ABC, abstractmethod
from mytypes import State, Action
from torch.optim.optimizer import Optimizer
from replay_buffer import ReplayBuffer
import torch
from network import Network
import torch.nn.functional as F

class Trainer(ABC):

    def __init__(self, optimizer: Optimizer, gamma: float) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.gamma = gamma

    @abstractmethod
    def train(self) -> torch.FloatTensor:
        raise NotImplementedError


class OffPolicyTrainer(Trainer):

    def __init__(
        self,
        policy_network: Network,
        target_network: Network,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.policy_network = policy_network
        self.target_network = target_network

    def train(self) -> dict[str, torch.FloatTensor]:
        if not self.replay_buffer.ready(self.batch_size):
            return torch.tensor([0])

        s, a, r, sn, d = self.replay_buffer.sample(self.batch_size)
        
        target = self.target(s, a)
        prediction = self.prediction(sn, r, d)
        losses = {}
        
        for dim in target.keys():
            loss = F.huber_loss(prediction[dim], target[dim])
            self.optimizer.zero_grad()
            for param in self.policy_network.parameters():
                param.clamp(-1, 1)
            loss.backward()
            losses[dim] = loss
        
        return losses

    def target(self, s: torch.Tensor, a: list[Action]) -> torch.Tensor:
        q = self.target_network(s)
        selected_qs = {}
        
        for action_dimension in q.keys():
            sub_q = q[action_dimension]
            sub_a = a[action_dimension]
            sub_a = sub_a.view((sub_a.size(0), -1))
            
            selected_q = sub_q.gather(1, sub_a)
            selected_qs[action_dimension] = selected_q.squeeze()
        
        return selected_qs

    def prediction(self, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        q = self.policy_network(sn)
        preds = {}
        not_done = d.logical_not()
        
        for action_dimension in q.keys():
            sub_q = q[action_dimension]
            max_q = sub_q.max(dim=1).values
            pred = r + not_done * self.gamma * max_q
            preds[action_dimension] = pred
        return preds
