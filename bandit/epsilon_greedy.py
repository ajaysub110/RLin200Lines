import numpy as np
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# global variables
n_bandits = 2000 # == n_runs
n_arms = 10
n_timesteps = 3000
sample_period = 4

# testbed
mean_rewards = np.array([np.random.normal(size=n_arms) for i in range(n_bandits)])

def get_actual_reward(bandit,arm):
    return np.random.normal(loc=mean_rewards[bandit,arm],scale=0.1)

# greedy
def greedy(epsilon=0, n_bandits=None, n_timesteps=None, Qt=None):
    rewards = []
    for i in range(n_bandits):
        bandit_rewards = []
        arm_count = np.zeros((n_arms))
        for j in range(n_timesteps):
            x = np.random.uniform()
            if x >= epsilon:
                at = np.argmax(Qt[i])
            else:
                at = np.random.randint(0,n_arms)
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

# run on varying epsilon values
for e in [0.1,0.0]:
    Qt = np.zeros((1,n_arms))
    Qt, rewards_exp = greedy(epsilon=1,n_bandits=1,n_timesteps=sample_period,Qt=Qt)

    Qt, rewards = greedy(epsilon=e,n_bandits=1,n_timesteps=n_timesteps,Qt=Qt)
    print(mean_rewards[0])
    print(Qt[0])

    combined_rewards = rewards_exp[0] + rewards[0]
    plt.plot(range(len(combined_rewards)),combined_rewards)
    
plt.title("epsilon 0.1 vs 0.0 for sample_period = 4, n_timesteps=2000 for reward variance 0.1")
plt.show()
