from .environment import Environment, State
import gym

class CartPole(Environment):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make("CartPole-v1")
        
    def step(self, action: int) -> tuple[State, float, bool]:
        _, r, done, _ = self.env.step(action)
        img = self.env.render("rgb_array")
        
        for prep in self.preps:
            img = prep(img)
        
        return img, r, done
    
    def reset(self) -> None:
        self.env.reset()
        
    def get_original_state_size(self) -> tuple[int]:
        self.reset()
        x = self.env.render("rgb_array")
        return x.shape

if __name__ == "__main__":
    from preprocessing import ToTensor
    preps = [
        ToTensor()
    ]
    cp = CartPole(preps)
    cp.reset()
    cp.step(1)
