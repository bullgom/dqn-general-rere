from environment.cartpole import CartPole
from network import CartPoleNetwork
from selection import EpsilonGreedySelection, LinearEpsilonGenerator
import preprocessing as prep
from agent import Agent
from replay_buffer import ReplayBuffer
from torch import optim
from trainer import OffPolicyTrainer
from plotter import Plotter, Plot
from saver import Saver
from collections import defaultdict
from datetime import datetime
import torch
import os
import wandb

wandb.login()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

buffer_capacity = 100000
lr = 0.001
batch_size = 64
gamma = 0.999
state_size = 84
eps_start = .9
eps_end = .1
eps_steps = 3000
num_frames = 4
switch_interval = 20

max_steps = 300000
max_steps_per_episode = 300
steps_per_train = 50
steps_per_report = 50
steps_per_save = 500

base_folder = "experiments"
run_name = datetime.now().strftime('%Y%d-%H%M%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

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
net = CartPoleNetwork(state_size, action_space).to(device)
egen = LinearEpsilonGenerator(eps_start, eps_end, eps_steps)
selection = EpsilonGreedySelection(egen, action_space)
agent = Agent(net, selection)
optimizer = optim.Adam(net.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_capacity, action_space, cpu, device)
trainer = OffPolicyTrainer(net, buffer, batch_size,
                           switch_interval, optimizer, gamma)

recorder = defaultdict(lambda: list())
LOSS = "loss"
DURATION = "duration"

plots = [
    Plot("Loss", f"steps ({steps_per_train})", "loss", {
        LOSS: recorder[LOSS]
    }),
    Plot("Duration", f"steps ({steps_per_train})", "duration", {
        DURATION: recorder[DURATION]
    })
]
plotter = Plotter(plots)
saver = Saver(
    base_folder, 
    run_name, 
    [
        "environment",
        "agent.py",
        "mytypes.py",
        "network.py",
        "plotter.py",
        "preprocessing.py",
        "recorder.py",
        "replay_buffer.py",
        "run.py",
        "saver.py",
        "selection.py",
        "trainer.py"
    ],
    {
        "policy.pt": net
    }
)
saver.save_experiment()

elapsed_steps = 0
best_duration = -1
while (elapsed_steps <= max_steps):
    s = env.reset()

    for i in range(max_steps_per_episode):
        elapsed_steps += 1
        a = agent.step(s.to(device))
        ns, r, d = env.step(a)
        buffer.append((s, a, r, ns, d))
        s = ns

        if (elapsed_steps % steps_per_train) == (steps_per_train - 1):
            losses = trainer.train()
            loss_sum = sum([loss.item()
                           for loss in losses.values()])/len(losses.values())
            if loss_sum != 0:
                recorder[LOSS].append(loss_sum)

        if (elapsed_steps % steps_per_report) == (steps_per_report - 1):
            plotter.plot(plots)
        
        if (elapsed_steps % steps_per_save) == (steps_per_save - 1):
            saver.save_state()

        if elapsed_steps > max_steps:
            break

        if d:
            break
    
    if i > best_duration:
        saver.results["best"] = i
        best_duration = i
    saver.results["last"] = i
    
    recorder[DURATION].append(i)
