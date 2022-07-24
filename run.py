from environment.cartpole import CartPole
from network import CartPoleNetwork
from selection import EpsilonGreedySelection, LinearEpsilonGenerator
import preprocessing as prep
from agent import Agent
from replay_buffer import ReplayBuffer
from torch import optim
from trainer import OffPolicyTrainer
from plotter import Plotter, Plot
from collections import defaultdict
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

buffer_capacity = 10000
lr = 0.001
batch_size = 5
gamma = 0.999
state_size = 84
eps_start = .9
eps_end = .1
eps_steps = 10000
num_frames = 3

max_steps = 10000
max_steps_per_episode = 300
steps_per_train = 100
steps_per_report = 1000

preps = [
    prep.ToTensor(),
    prep.AddBatchDim(),
    prep.Resize({"w": state_size, "h": state_size}),
    prep.Grayscale(),
    prep.MultiFrame(num_frames)
]

env = CartPole(preps)
state_size = env.state_size()
action_space = env.action_space()
net = CartPoleNetwork(state_size, action_space)
egen = LinearEpsilonGenerator(eps_start, eps_end, eps_steps)
selection = EpsilonGreedySelection(egen, action_space)
agent = Agent(net, selection)
optimizer = optim.Adam(net.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_capacity, action_space)
trainer = OffPolicyTrainer(net, net.copy(), buffer, batch_size, optimizer, gamma)

recorder = defaultdict(lambda : list())
LOSS = "loss"

plots = [
    Plot("Loss", f"steps ({steps_per_train})", "loss", {
        LOSS: recorder[LOSS]
    })
]
plotter = Plotter(plots)


elapsed_steps = 0
while (elapsed_steps <= max_steps):
    s = env.reset()
    
    for i in range(max_steps_per_episode):
        elapsed_steps += 1
        a = agent.step(s)
        ns, r, d = env.step(a)
        buffer.append((s, a, r, ns, d))
        s = ns
        
        if (elapsed_steps % steps_per_train) == (steps_per_train - 1):
            losses = trainer.train()
            loss_sum = sum([loss.item() for loss in losses.values()])/len(losses.values())
            recorder[LOSS].append(loss_sum)
            
        if (elapsed_steps % steps_per_report) == (steps_per_report - 1):
            plotter.plot(plots)
        
        if elapsed_steps > max_steps:
            break
        
        if d:
            break

