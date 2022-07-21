from environment.environment import Environment, State
import gym
from mytypes import Size


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

    def original_state_size(self) -> tuple[int]:
        self.reset()
        x = self.env.render("rgb_array")
        s = x.shape
        x = {"w": s[1], "h": s[0], "c": s[2]}
        return x

    def output_size(self) -> Size:
        return {"output": self.env.action_space.n}


if __name__ == "__main__":
    from preprocessing import ToTensor, Resize
    preps = [
        ToTensor(),
        Resize({"w": 84, "h": 84})
    ]
    cp = CartPole(preps)
    cp.reset()
    state, r, done = cp.step(1)
    a = 1
