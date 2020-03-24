"""
Vanilla Policy Gradient

STEPS:
1. Imports: autograd.Variable, gym
2. Make env, set seeds for env and torch
3. Hyperparameters: lr=0.01, gamma=0.99
4. PolicyNet: n_h=128, book keeping stuff within
5. Instantiate policy and optimizer
6. select_action: stochastic policy using Categorical
7. update_policy: calculate returns, scale returns, calculate loss, update network
8. training loop: 
    for each episode (1000):
        for each timestep (1000):
            select action
            take action
            save reward
            update policy
            book keeping
9. plot results
"""

# TODO: replace reward with advantage estimate
# TODO: batch of trajectories

# 1
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as opt 
from torch.autograd import Variable
from torch.distributions import Categorical
import gym
from torch.utils.tensorboard import SummaryWriter

# 2
env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

# 3
lr_p = 0.01
lr_v = 0.0005
gamma = 0.99

# 4
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet,self).__init__()

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.fc1 = nn.Linear(self.state_space,128)
        self.fc2 = nn.Linear(128,self.action_space)

        self.gamma = gamma
        self.policy_hist = Variable(torch.Tensor())
        self.episode_reward = []
        self.loss_hist = []
        self.reward_hist = []

    def forward(self, x):
        model = nn.Sequential(
            self.fc1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.fc2,
            nn.Softmax(dim=-1)
        )

        return model(x)

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.state_space = env.observation_space.shape[0]

        self.fc1 = nn.Linear(self.state_space,32)
        self.fc2 = nn.Linear(32,1)

        self.value_hist = Variable(torch.Tensor())
        self.loss_hist = []

    def forward(self, x):
        model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.6),
            self.fc2
        )

        return model(x)

# 5
policy = PolicyNet()
value = ValueNet()
optimizer_policy = opt.Adam(policy.parameters(),lr=lr_p)
optimizer_value = opt.Adam(value.parameters(),lr=lr_v)
tb = True
if tb:
    writer = SummaryWriter()

# 6
def select_action(s):
    state = torch.from_numpy(s).type(torch.FloatTensor)
    
    probs = policy(Variable(state))
    val = value(Variable(state))
    c = Categorical(probs=probs)
    action = c.sample()
    
    policy.policy_hist = torch.cat([
        policy.policy_hist,c.log_prob(action).unsqueeze(0)])
    value.value_hist = torch.cat([
        value.value_hist,val])

    return action

# 7
def update_policy(ep):
    returns = []
    R = 0

    for r in policy.episode_reward[::-1]:
        R = r + policy.gamma * R 
        returns.insert(0,R)

    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (
        returns.std() + np.finfo(np.float32).eps)

    loss_policy = torch.sum(torch.mul(
        policy.policy_hist,Variable(returns)-Variable(value.value_hist)
        ).mul(-1), -1).unsqueeze(0)
    writer.add_scalar('loss/policy',loss_policy.item(),ep)

    loss_value = nn.MSELoss()(value.value_hist, returns).unsqueeze(0)
    writer.add_scalar('loss/value',loss_value.item(),ep)

    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    policy.loss_hist.append(loss_policy.item())
    value.loss_hist.append(loss_value.item())
    policy.reward_hist.append(np.sum(policy.episode_reward))
    policy.policy_hist = Variable(torch.Tensor())
    value.value_hist = Variable(torch.Tensor())
    policy.episode_reward = []

# 8
def main():
    for ep in range(1000):
        s = env.reset()
        done = False
        for t in range(1000):
            a = select_action(s)
            s, r, done, _ = env.step(a.item())

            policy.episode_reward.append(r)

            if done:
                break

        episode_reward = np.sum(policy.episode_reward)
        update_policy(ep)

        if ep % 20 == 0 and tb == True:
            print("Episode: {}, reward: {}, duration: {}".format(
                ep, episode_reward,t
            ))
            writer.add_scalar('reward',episode_reward,ep)

main()
if tb:
    writer.close()