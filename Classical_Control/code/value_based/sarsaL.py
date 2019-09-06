import gym
import gym_puddle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Hyperparameters
N_RUNS = 25
N_EPISODES = 500
MAX_STEPS = 1000
GAMMA = 0.9
EPSILON = 0.2
EPSILON_DECAY = 1.0
ALPHA = 0.5
LAMBDA = [0,0.3,0.5,0.9,0.99,1.0]

def get_index(state):
    return state[0]*12 + state[1]

env = gym.make('puddleC-v0')

N_STATES = env.height * env.width # 144

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

for lamb in LAMBDA:
    rewards = np.zeros((N_RUNS,N_EPISODES))
    steps_to_goal = np.zeros((N_RUNS,N_EPISODES))
    for k in range(N_RUNS):
        q_table = np.random.standard_normal((N_STATES,env.action_space.n))
        e_table = np.zeros((N_STATES,env.action_space.n))
        print("RUN"+str(k))
        for i in range(N_EPISODES):
            state = env.reset()

            for j in range(MAX_STEPS):
                if np.random.uniform(0, 1) < EPSILON:
                    action = env.action_space.sample() # 2
                else:
                    action = np.argmax(q_table[get_index(state),:])

                next_state, reward, ended, _ = env.step(action)

                if np.random.uniform(0, 1) < EPSILON:
                    next_action = env.action_space.sample() # 2
                else:
                    next_action = np.argmax(q_table[get_index(next_state),:])

                delta = reward + GAMMA*q_table[get_index(next_state),next_action]-q_table[get_index(state),action]
                e_table[get_index(state),action] += 1

                for s in range(N_STATES):
                    for a in range(env.action_space.n):
                        q_table[s,a] += ALPHA*delta*e_table[s,a]
                        e_table[s,a] *= GAMMA*lamb

                # print("Run: {}, Episode: {}, Step: {}, State: {}, Action: {}, Reward: {}, ended: {}".format(k,i,j,state,action,reward,ended))

                state = next_state
                rewards[k,i] += reward

                if ended == True:
                    break 
            steps_to_goal[k,i] = j+1
            EPSILON *= EPSILON_DECAY
        print()
    print()

    average_rewards = np.mean(rewards,axis=0)
    average_steps_to_goal = np.mean(steps_to_goal,axis=0)

    ax1.plot(range(N_EPISODES),average_rewards, label=str(lamb))
    ax1.legend()
    ax2.plot(range(N_EPISODES),average_steps_to_goal)
    ax2.legend()

plt.show()