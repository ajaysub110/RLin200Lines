# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt 
from torch.utils.tensorboard import SummaryWriter
import gym
from copy import deepcopy
from collections import deque
import random

# hyperparameters
seed = 0
replay_size = 1000000
batch_size = 100
gamma = 0.99
lr_p = 0.001
lr_q = 0.001
polyak = 0.995
epochs = 100
start_steps = 10000
steps_per_epoch = 4000
noise_std = 0.1
max_ep_len = 1000
update_after = 1000
update_every = 50
tb = True

# set seeds
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# define env
env = gym.make('Pendulum-v0')
state_space = env.observation_space.shape
action_space = env.action_space.shape[0]
action_limit = env.action_space.high[0]

# define actor critic model
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_limit):
        super().__init__()

        self.a_limit = a_limit

        self.fc1 = nn.Linear(s_dim, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32, a_dim)

    def forward(self, x):
        model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.Tanh()
        )

        return self.a_limit * model(x)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, a_limit):
        super().__init__()

        self.a_limit = a_limit

        self.fc1 = nn.Linear(s_dim+a_dim, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, s, a):
        model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

        return torch.squeeze(model(torch.cat([s,a],dim=-1)),-1) # TODO: squeeze


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()

        s_dim = observation_space.shape[0]
        a_dim = action_space.shape[0]
        a_limit = action_space.high[0]

        self.p = Actor(s_dim, a_dim, a_limit)
        self.q = Critic(s_dim, a_dim, a_limit)

    def select_action(self, s):
        with torch.no_grad():
            return self.p(s).numpy()

agent = ActorCritic(env.observation_space, env.action_space)
agent_targ = deepcopy(agent)

for p in agent_targ.parameters():
    p.requires_grad = False

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)

    def push(self, x):
        self.memory.append(x)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.as_tensor(v,dtype=torch.float32) for v in [state, action, reward, next_state, done])

    def get_len(self):
        return len(self.memory)

replay_buffer = ReplayBuffer(replay_size)

def get_q_loss(s,a,r,s1,d):
    q = agent.q(s,a)

    with torch.no_grad():
        q_pi_targ = agent_targ.q(s1, agent_targ.p(s1))
        target = r + gamma * (1 - d) * q_pi_targ
    
    loss_q = nn.MSELoss()(q, target)

    return loss_q 

def get_p_loss(s):
    q_pi = agent.q(s, agent.p(s))
    return -torch.mean(q_pi)

p_optimizer = opt.Adam(agent.p.parameters(),lr=lr_p)
q_optimizer = opt.Adam(agent.q.parameters(), lr=lr_q)

def update_params(s,a,r,s1,d):
    q_optimizer.zero_grad()
    loss_q = get_q_loss(s,a,r,s1,d)
    loss_q.backward()
    q_optimizer.step()

    for p in agent.q.parameters():
        p.requires_grad = False

    p_optimizer.zero_grad()
    loss_p = get_p_loss(s)
    loss_p.backward()
    p_optimizer.step()

    for p in agent.q.parameters():
        p.requires_grad = True

    with torch.no_grad():
        for p, p_targ in zip(agent.parameters(),agent_targ.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1-polyak)*p.data)

def get_action(s, noise):
    a = agent.select_action(torch.as_tensor(s,dtype=torch.float32))
    a += noise * np.random.randn(action_space)
    return np.clip(a, -action_limit, action_limit)

if tb:
    writer = SummaryWriter()
total_steps = steps_per_epoch * epochs
s, ep_r, ep_len, ep = env.reset(), 0, 0, 0

for t in range(total_steps):
    # select action
    if t > start_steps:
        a = get_action(s, noise_std)
    else:
        a = env.action_space.sample()

    # take a step
    s1, r, d, _ = env.step(a)
    ep_r += r
    ep_len += 1

    d = False if ep_len == max_ep_len else d

    replay_buffer.push((s,a,r,s1,d))

    s = s1 

    # if done
    if d or (ep_len == max_ep_len):
        if ep % 20 == 0:
            print("Ep: {}, reward: {}, t: {}".format(ep, ep_r, t))
        if tb:
            writer.add_scalar('episode_reward',ep_r,t)
            
        s, ep_r, ep_len = env.reset(), 0, 0
        ep += 1

    # if update time
    if t >= update_after and t % update_every == 0:
        for _ in range(update_every):
            s_b, a_b, r_b, s1_b, d_b = replay_buffer.sample(batch_size)
            update_params(s_b, a_b, r_b, s1_b, d_b)

if tb:
    writer.close()
env.close()