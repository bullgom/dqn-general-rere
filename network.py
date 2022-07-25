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

    def to(self, device: torch.device) -> "Network":
        super().to(device)
        self.device = device
        return self
    
class CartPoleNetwork(Network):
    
    def __init__(self, size: Size, action_space: ActionSpace) -> None:
        super().__init__()
        self.size = size
        self.action_space = action_space

        w, h, c = size["w"], size["h"], size["c"]
        
        ch1 = 32
        self.conv1 = nn.Conv2d(c, ch1, 5, 2)
        self.bn1 = nn.BatchNorm2d(ch1)
        w = self.calculate_conv_size(w, 5, 2)
        h = self.calculate_conv_size(h, 5, 2)
        ch2 = 32
        self.conv2 = nn.Conv2d(ch1, ch2, 5, 2)
        self.bn2 = nn.BatchNorm2d(ch2)
        w = self.calculate_conv_size(w, 5, 2)
        h = self.calculate_conv_size(h, 5, 2)
        ch3 = 32
        self.conv3 = nn.Conv2d(ch2, ch3, 5, 2)
        self.bn3 = nn.BatchNorm2d(ch3)
        w = self.calculate_conv_size(w, 5, 2)
        h = self.calculate_conv_size(h, 5, 2)
        
        linear_size = w * h * ch3
        
        self.fc = nn.Linear(linear_size, action_space["move_direction"])
    
    def calculate_conv_size(self, size: int, kernel_size: int, stride: int) -> int:
        return (size - (kernel_size - 1) - 1)// stride + 1

    def __call__(self, x: State) -> Q:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x)
        
        return {"move_direction": x}
    
    def copy(self) -> "CartPoleNetwork":
        net = CartPoleNetwork(self.size, self.action_space)
        net.load_state_dict(self.state_dict())
        
        if self.device:
            net = net.to(self.device)
        
        return net
    

