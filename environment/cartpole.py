from environment.environment import Environment
import gym
from mytypes import Size, State, ActionSpace, Action


class CartPole(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = gym.make("CartPole-v1")

    def state(self) -> State:
        img = self.env.render(mode="rgb_array")
        for prep in self.preps:
            img = prep(img)
        return img

    def step(self, action: Action) -> tuple[State, float, bool]:
        _, r, done, _ = self.env.step(action["move_direction"])
        img = self.state()
        return img, r, done

    def reset(self) -> State:
        self.env.reset()
        
        for prep in self.preps:
            prep.reset()
        
        state = self.state()
        return state

    def original_state_size(self) -> Size:
        self.reset()
        x = self.env.render(mode="rgb_array")
        s = x.shape
        x = {"w": s[1], "h": s[0], "c": s[2]}
        return x

    def action_space(self) -> ActionSpace:
        return {"move_direction": self.env.action_space.n}
