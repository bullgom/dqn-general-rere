import torch
import torch.nn as nn
import torch.nn.functional as F
from mytypes import State, Size, Q, ActionSpace
from typing import overload
from abc import ABC, abstractmethod

class Network(torch.nn.Module):
    @overload
    def __call__(self, x: State) -> Q:
        pass
    
    @abstractmethod
    def copy(self) -> "Network":
        raise NotImplementedError

    
class CartPoleNetwork(Network):
    
    def __init__(self, size: Size, action_space: ActionSpace) -> None:
        super().__init__()
        self.size = size
        self.action_space = action_space

        w, h, c = size["w"], size["h"], size["c"]
        
        ch1 = 32
        self.conv1 = nn.Conv2d(c, ch1, 3, 2)
        w = self.calculate_conv_size(w, 3, 2)
        h = self.calculate_conv_size(h, 3, 2)
        ch2 = 32
        self.conv2 = nn.Conv2d(ch1, ch2, 3, 2)
        w = self.calculate_conv_size(w, 3, 2)
        h = self.calculate_conv_size(h, 3, 2)
        ch3 = 32
        self.conv3 = nn.Conv2d(ch2, ch3, 3, 2)
        w = self.calculate_conv_size(w, 3, 2)
        h = self.calculate_conv_size(h, 3, 2)
        
        linear_size = w * h * ch3
        
        l = 100
        self.fc = nn.Linear(linear_size, l)
        self.fc2 = nn.Linear(l, action_space["move_direction"])
    
    def calculate_conv_size(self, size: int, kernel_size: int, stride: int) -> int:
        return int((size - kernel_size//2)/stride)

    def __call__(self, x: State) -> Q:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view((1, -1))
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return {"move_direction": x}
    
    def copy(self) -> "CartPoleNetwork":
        net = CartPoleNetwork(self.size, self.action_space)
        net.load_state_dict(self.state_dict())
        return net
