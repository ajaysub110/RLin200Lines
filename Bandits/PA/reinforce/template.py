import numpy as np
from ads import UserAdvert
import matplotlib.pyplot as plt

ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10
STEP_SIZE = 0.01


def learnBandit():
    env = UserAdvert()
    rew_vec = []

    W = np.random.randn(4,3)

    for train_step in range(TRAIN_STEPS):
        state = env.getState()
        stateVec = state["stateVec"]
        stateId = state["stateId"]

        # ---- UPDATE code below ------j
        # Sample from policy = softmax(stateVec X W) [W learnable params]
        # policy = function (stateVec)
        policy = softmax(np.dot(stateVec.T,W))
        action = int(np.random.choice(range(3),p=policy))
        reward = env.getReward(stateId, action)
        # ----------------------------

        # ---- UPDATE code below ------
        # Update policy using reward
        W += STEP_SIZE * reward * np.dot(
            stateVec.reshape(STATE_SIZE,1),policy.reshape((1,ACTION_SIZE))-1)
        # ----------------------------

        if train_step % LOG_INTERVAL == 0:
            print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------
                # Policy = function(testV)
                policy = softmax(np.dot(testV.T,W))
                # ----------------------------
                act = int(np.random.choice(range(3), p=policy))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    # Plot this rew_vec list
    # print(rew_vec)
    plt.plot(range(0,TRAIN_STEPS,LOG_INTERVAL),rew_vec)
    plt.show()

def softmax(x):
    num = np.exp(x)
    return num / np.sum(num)


if __name__ == '__main__':
    learnBandit()
