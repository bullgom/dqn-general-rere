from environment.ale import ALE
from network import ALENetwork
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
import gym

def interval(t: int, interval: int) -> bool:
    return (t % interval) == (interval - 1)

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
max_steps_per_episode = 5000
steps_per_train = 50
steps_per_report = 50
steps_per_save = 500

base_folder = "experiments"
run_name = datetime.now().strftime('%Y%d-%H%M%S')
plot_path = os.path.join(base_folder, run_name, "plot.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

preps = [
    prep.ToTensor(),
    prep.AddBatchDim(),
    prep.Resize({"w": state_size, "h": state_size}),
    prep.Grayscale(),
    prep.MultiFrame(num_frames)
]

inner_env = gym.make("ALE/Breakout-v5", frameskip=3)
env = ALE(inner_env, preps)
state_size = env.state_size()
action_space = env.action_space()
net = ALENetwork(state_size, action_space).to(device)
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
EPISODE_REWARD = "ep_reward"

plots = [
    Plot("Loss", f"steps ({steps_per_train})", "loss", {
        LOSS: recorder[LOSS]
    }),
    Plot("Duration", f"steps ({steps_per_train})", "duration", {
        DURATION: recorder[DURATION]
    }),
    Plot("Reward Per Episode", f"episodes", "reward", {
        EPISODE_REWARD: recorder[EPISODE_REWARD]
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
    r_sum = 0
    for i in range(max_steps_per_episode):
        elapsed_steps += 1
        a = agent.step(s.to(device))
        ns, r, d = env.step(a)
        buffer.append((s, a, r, ns, d))
        s = ns
        r_sum += r

        if interval(elapsed_steps, steps_per_train):
            losses = trainer.train()
            loss_sum = sum([loss.item()
                           for loss in losses.values()])/len(losses.values())
            if loss_sum != 0:
                recorder[LOSS].append(loss_sum)

        if interval(elapsed_steps, steps_per_report):
            plotter.plot(plots)
            plotter.show()
            plotter.save(plot_path)
        
        if interval(elapsed_steps, steps_per_save):
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
    recorder[EPISODE_REWARD].append(r_sum)
