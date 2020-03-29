"""
Proximal Policy Optimization
PyTorch implementation of Proximal Policy Optimization with clipping loss and
deep networks for policy and value function parameters.
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
from torch.distributions import Categorical
import gym
from torch.utils.tensorboard import SummaryWriter

# Create env and set env and torch seeds
env = gym.make("CartPole-v1")
env.seed(42)
torch.manual_seed(42)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Training hyperparameters
lr_p = 0.001
lr_v = 0.005
gamma = 0.99
batch_size = 8
epsilon = 0.2
copy_interval = 20


# Policy network
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.fc1 = nn.Linear(self.state_space, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.action_space)

        self.gamma = gamma
        self.policy_hist = Variable(torch.Tensor())
        self.traj_reward = []
        self.loss_hist = Variable(torch.Tensor())

    def forward(self, x):
        model = nn.Sequential(
            self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3,
            nn.Softmax(dim=-1)
        )

        return model(x)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.state_space = env.observation_space.shape[0]

        self.fc1 = nn.Linear(self.state_space, 24)
        self.fc2 = nn.Linear(24, 1)

        self.value_hist = Variable(torch.Tensor())
        self.loss_hist = Variable(torch.Tensor())

    def forward(self, x):
        model = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

        return model(x)


# Instantiate networks and optimizers
policy_new, policy_old = PolicyNet().to(device), PolicyNet().to(device)
policy_old.load_state_dict(policy_new.state_dict())
value = ValueNet().to(device)
optimizer_policy = opt.Adam(policy_new.parameters(), lr=lr_p)
optimizer_value = opt.Adam(value.parameters(), lr=lr_v)
tb = False
if tb:
    writer = SummaryWriter()


# stochastic policy
def select_action(s):
    state = torch.from_numpy(s).type(torch.FloatTensor).to(device)

    probs_old = policy_old(Variable(state))
    probs_new = policy_new(Variable(state))
    val = value(Variable(state))
    c = Categorical(probs=probs_old)
    action = c.sample()

    policy_old.policy_hist = torch.cat(
        [policy_old.policy_hist, probs_old[action].unsqueeze(0)]
    )

    policy_new.policy_hist = torch.cat(
        [policy_new.policy_hist, probs_new[action].unsqueeze(0)]
    )

    value.value_hist = torch.cat([value.value_hist, val])

    return action


# compute losses for current trajectories
def get_traj_loss():
    R = 0
    returns = []

    for r in policy_old.traj_reward[::-1]:
        R = r + policy_old.gamma * R
        returns.insert(0, R)

    returns = torch.FloatTensor(returns).to(device)
    A = Variable(returns) - Variable(value.value_hist)
    ratio = torch.div(policy_new.policy_hist, policy_old.policy_hist)
    clipping = torch.clamp(ratio, 1 - epsilon, 1 + epsilon).mul(A).to(device)

    loss_policy = (
        torch.mean(torch.min(torch.mul(ratio, A), clipping)).mul(
            -1).unsqueeze(0)
    )
    loss_value = nn.MSELoss()(value.value_hist, Variable(returns)).unsqueeze(0)

    policy_new.loss_hist = torch.cat([policy_new.loss_hist, loss_policy])
    value.loss_hist = torch.cat([value.loss_hist, loss_value])

    policy_old.traj_reward = []
    policy_old.policy_hist = Variable(torch.Tensor())
    policy_new.policy_hist = Variable(torch.Tensor())
    value.value_hist = Variable(torch.Tensor())


# network update after every batch of trajectories (end of episode)
def update_policy(ep):
    loss_policy = torch.mean(policy_new.loss_hist)
    loss_value = torch.mean(value.loss_hist)

    if tb:
        writer.add_scalar("loss/policy", loss_policy, ep)
        writer.add_scalar("loss/value", loss_value, ep)

    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    policy_new.loss_hist = Variable(torch.Tensor())
    value.loss_hist = Variable(torch.Tensor())


def main():
    for ep in range(1000):
        episode_reward = 0
        for i in range(batch_size):
            s = env.reset()
            done = False
            for t in range(1000):
                a = select_action(s)
                s, r, done, _ = env.step(a.item())

                policy_old.traj_reward.append(r)

                if done:
                    break

            episode_reward += np.sum(policy_old.traj_reward) / batch_size
            get_traj_loss()

        update_policy(ep)

        if ep % 20 == 0:
            print("Episode: {}, reward: {}".format(ep, episode_reward))
            if tb:
                writer.add_scalar("reward", episode_reward, ep)

        if ep % copy_interval == 0:
            policy_old.load_state_dict(policy_new.state_dict())


main()
if tb:
    writer.close()
