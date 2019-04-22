import numpy as np
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# global variables
n_bandits = 2000 # == n_runs
n_arms = 10
n_timesteps = 3000

# testbed
mean_rewards = np.array([np.random.normal(size=n_arms) for i in range(n_bandits)])

def get_actual_reward(bandit,arm):
    return np.random.normal(loc=mean_rewards[bandit,arm],scale=0.1)

# print(get_actual_reward(1,2))

def softmax(x, beta):
    e_x = np.exp(x/beta)
    return e_x / np.sum(e_x)

# print(softmax(np.array([2,3,4,5]),1))

def train(beta, n_bandits=None, n_timesteps=None, Qt=None):
    rewards = []
    for i in range(n_bandits):
        bandit_rewards = []
        arm_count = np.zeros((n_arms))
        for j in range(n_timesteps):
            probs = softmax(Qt[i],beta)
            at = np.random.choice(range(n_arms),p=probs)

            arm_count[at] += 1
            Rt = get_actual_reward(i,at)

            if j%3 == 0:
                bandit_rewards.append(Rt)

            Qt[i,at] = Qt[i,at] + (Rt - Qt[i,at])/(arm_count[at] + 1)

            if j%50 == 0:
                print("Bandit: {} Timestep: {} Action: {} Reward: {}".format(i,j,at,Rt))
        rewards.append(bandit_rewards)
        print()
    
    return Qt, rewards

Qt = np.zeros((1,n_arms))
Qt, rewards_1 = train(beta=1,n_bandits=1,n_timesteps=500,Qt=Qt)

print(mean_rewards[0])
print(Qt[0])

Qt, rewards_2 = train(beta=0.01,n_bandits=1,n_timesteps=n_timesteps,Qt=Qt)

plt.plot(range(len(rewards_1[0] + rewards_2[0])),rewards_1[0] + rewards_2[0])
plt.title("beta=1 for n_timesteps=3000 for true reward variance 0.1")
plt.show()