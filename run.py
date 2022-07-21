from environment.cartpole import CartPole
from network import CartPoleNetwork
from selection import EpsilonGreedySelection, LinearEpsilonGenerator
import preprocessing as prep
from agent import Agent

preps = [
    prep.ToTensor(),
    prep.Resize({"w": 84, "h": 84})
]

env = CartPole(preps)
state_size = env.state_size()
action_space = env.action_space()
net = CartPoleNetwork(state_size, action_space)
egen = LinearEpsilonGenerator(.9, .1, 33)
selection = EpsilonGreedySelection(egen, action_space)
agent = Agent(net, selection)

state = env.reset()
action = agent.step(state)
ns, r, d = env.step(action)
action = agent.step(ns)
ns, r, d = env.step(action)
