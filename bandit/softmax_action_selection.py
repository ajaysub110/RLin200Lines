import numpy as np
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# global variables
n_bandits = 500
n_arms = 50
n_timesteps = 1000

# testbed
mean_rewards = np.array([np.random.normal(size=n_arms) for i in range(n_bandits)])

def get_actual_reward_list(bandit):
    reward_list = []
    for a in range(n_arms):
        reward_list.append(np.random.normal(loc=mean_rewards[bandit,a],scale=1.0))
    
    return np.array(reward_list)

# print(get_actual_reward(1,2))

def softmax(x, beta):
    e_x = np.exp(x/beta)
    return e_x / np.sum(e_x)

# print(softmax(np.array([2,3,4,5]),1))

def train(beta, n_bandits=None, n_timesteps=None):
    rewards = []
    optimal_action = []
    Qt = np.zeros((n_bandits,n_arms))
    for i in range(n_bandits):
        bandit_rewards = []
        bandit_optimal_action = []        
        arm_count = np.zeros((n_arms))
        for j in range(n_timesteps):
            probs = softmax(Qt[i],beta)
            at = np.random.choice(range(n_arms),p=probs)

            arm_count[at] += 1
            actual_reward_list = get_actual_reward_list(i)
            Rt = actual_reward_list[at]

            bandit_rewards.append(Rt)

            if at == np.argmax(actual_reward_list):
                bandit_optimal_action.append(1.0)
            else:
                bandit_optimal_action.append(0.0)

            Qt[i,at] = Qt[i,at] + (Rt - Qt[i,at])/(arm_count[at] + 1)

            if j%500==499:
                print("Bandit: {} Timestep: {} Action: {} Reward: {}".format(i,j,at,Rt))
        rewards.append(bandit_rewards)
        optimal_action.append(bandit_optimal_action)
        print()
    
    return Qt, rewards, optimal_action

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

for beta in [0.1]:
    Qt, rewards, optimal_action = train(beta=beta,n_bandits=n_bandits,n_timesteps=n_timesteps)
    rewards = np.array(rewards)
    optimal_action = np.array(optimal_action)
    percentages = (np.sum(optimal_action,0) * 100)/optimal_action.shape[0]

    print(mean_rewards[0])
    print(Qt[0])

    ax1.plot(range(len(np.mean(rewards,0))),np.mean(rewards,0))
    ax2.plot(range(len(percentages)),percentages)

plt.show()