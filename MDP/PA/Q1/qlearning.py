import gym
import gym_puddle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Hyperparameters
N_RUNS = 50
N_EPISODES = 200
MAX_STEPS = 1000
GAMMA = 0.9
EPSILON = 0.2
EPSILON_DECAY = 1.0
ALPHA = 0.5

def get_index(state):
    return state[0]*12 + state[1]

env = gym.make('puddleC-v0')

N_STATES = env.height * env.width # 144

rewards = np.zeros((N_RUNS,N_EPISODES))
steps_to_goal = np.zeros((N_RUNS,N_EPISODES))

for k in range(N_RUNS): # Run
    q_table = np.random.standard_normal((N_STATES,env.action_space.n)) # (144,4)
    for i in range(N_EPISODES): # Episode
        state = env.reset() # (5,3)
    
        for j in range(MAX_STEPS): # Step
            if np.random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample() # 2
            else:
                action = np.argmax(q_table[get_index(state),:])

            next_state, reward, ended, _ = env.step(action)

            update_term =reward+GAMMA*np.max(q_table[get_index(next_state),:])-q_table[get_index(state),action]
            q_table[get_index(state),action] += ALPHA * update_term

            print("Run: {}, Episode: {}, Step: {}, State: {}, Action: {}, Reward: {}, ended: {}".format(k,i,j,state,action,reward,ended))

            state = next_state
            rewards[k,i] += reward

            if ended == True:
                break
        
        steps_to_goal[k,i] = j+1

        EPSILON *= EPSILON_DECAY
        print()
    print()

print(q_table)

# Plot learning curves
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
average_rewards = np.mean(rewards,axis=0)
average_steps_to_goal = np.mean(steps_to_goal,axis=0)

print(average_rewards)
print(average_steps_to_goal)

ax1.plot(range(N_EPISODES),average_rewards)
ax2.plot(range(N_EPISODES),average_steps_to_goal)

plt.show()
