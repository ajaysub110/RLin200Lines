import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_UPDATE_INTERVAL = 100
TENSORBOARD_LOG = False
TB_LOG_PATH = "./runs/dqn/run2"
REPLAY_BUFFER_CAPACITY = 2000
env = gym.make("CartPole-v0")
# env = env.unwrapped
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
N_EPISODES = 1000

# helpers
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.1)


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)

    def push(self, x):
        self.memory.append(x)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def get_len(self):
        return len(self.memory)


# network definition
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(STATE_DIM, 50)
        self.fc2 = nn.Linear(50, ACTION_DIM)

        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Agent(object):
    def __init__(self):
        self.dqn, self.target_dqn = DQN(), DQN()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.optimizer = opt.Adam(self.dqn.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def get_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)

        if np.random.uniform() < EPSILON:
            qs = self.dqn.forward(s)
            action = torch.max(qs, 1)[1].data.numpy()
            action = action[0]
        else:
            action = env.action_space.sample()

        return action

    def update_params(self):
        # update target network
        if self.learn_step_counter % TARGET_UPDATE_INTERVAL == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.learn_step_counter += 1

        # sample batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions.astype(int).reshape((-1, 1)))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

        # get q values
        q_current = self.dqn(states).gather(1, actions)
        q_next = self.target_dqn(next_states).detach()
        q_target = rewards + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_loss = self.loss_fn(q_current, q_target)

        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()


# create agent
agent = Agent()
if TENSORBOARD_LOG:
    writer = SummaryWriter(TB_LOG_PATH)

print("\nCollecting experience")
for ep in range(400):
    state = env.reset()
    episode_reward = 0
    step = 0

    while True:
        # env.render()
        action = agent.get_action(state)

        # take action
        next_state, reward_orig, done, _ = env.step(action)
        step += 1

        # modify the reward
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (
            env.theta_threshold_radians - abs(theta)
        ) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        agent.replay_buffer.push((state, action, reward, next_state, done))
        agent.memory_counter += 1

        episode_reward += reward_orig

        if agent.memory_counter > REPLAY_BUFFER_CAPACITY:
            agent.update_params()

            if done:
                print(
                    "Episode: {}, Reward: {}, step: {}".format(
                        ep, round(episode_reward, 2), step
                    )
                )

        if done:
            break

        state = next_state
    if TENSORBOARD_LOG:
        writer.add_scalar("episode_reward", episode_reward, ep)
env.close()
