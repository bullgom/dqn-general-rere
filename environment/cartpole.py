from .environment import Environment
import gym

class CartPole(Environment):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_ = gym.make("CartPole-v1")
        
    def step(self, action: int):
        _, r, done, _ = self.env_.step(action)
        self.env_.render("")
        
        
    