import numpy as np


class UserAdvert(object):
    """
    Contextual bandit environment:

    There are 3 types of advertisements and each user is represented by a
    vector. Your task is to build a contextual bandit that gives an appropriate
    action (suggests appropriate add for each user)

    The reward is the profit (in $), as a result of playing that advertisement.

    check sample.py to see how to use function
    """
    def __init__(self):
        # Set random seed
        np.random.seed(100)

        # Load data and normalize
        self.data = np.loadtxt("ads.csv", delimiter=",")
        np.random.shuffle(self.data)

        self.labels = self.data[:, 4]
        self.data = self.data[:, :4]
        self.data = self.data - self.data.mean(axis=0)
        self.data = self.data / self.data.std(axis=0)

        # Set internal variables
        self.counter = 0
        self.num = self.data.shape[0]
        self.means = [[3, 1, 1],
                      [1, 3, 1],
                      [1, 1, 3]]
        self.var = 1.0

    def getState(self):
        self.counter = (self.counter + 1) % self.num
        curData = self.data[self.counter]
        returnObject = {
            "stateVec": curData,
            "stateId": self.counter
        }
        return returnObject

    def getReward(self, stateId, action):
        """
        Get reward for performing 'action' on 'stateId'
        """
        assert(action in [0, 1, 2] and type(action) is int), \
            "Invalid action, action must be an int which is 0, 1 or 2"
        #  Add 0.2 to avoid rounding errors
        dataClass = int((self.labels[stateId]) + 0.2)
        reward = np.random.normal(self.means[dataClass][action], self.var),
        return reward[0]
