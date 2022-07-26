from environment.environment import Environment
import gym
from mytypes import Size, State, ActionSpace, Action


class ALE(Environment):
    ACTION = "ACTION"

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.min_reward = float("inf")
        self.max_reward = -float("inf")

    def state(self) -> State:
        img = self.env.render(mode="rgb_array")
        for prep in self.preps:
            img = prep(img)
        return img

    def step(self, action: Action) -> tuple[State, float, bool]:
        _, r, done, _ = self.env.step(action[self.ACTION])
        img = self.state()

        r = self.clip_reward(r)
        
        return img, r, done

    def reset(self) -> State:
        self.env.reset()
        
        for prep in self.preps:
            prep.reset()
        
        state = self.state()
        return state
    
    def clip_reward(self, r: float) -> float:
        if r > 1:
            r = 1
        elif r < -1:
            r = -1
        
        return r

    def original_state_size(self) -> Size:
        self.reset()
        x = self.env.render(mode="rgb_array")
        s = x.shape
        x = {"w": s[1], "h": s[0], "c": s[2]}
        return x

    def action_space(self) -> ActionSpace:
        return {self.ACTION: self.env.action_space.n}
