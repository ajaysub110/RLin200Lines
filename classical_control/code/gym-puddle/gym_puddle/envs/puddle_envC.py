import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random

class PuddleEnvC(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.height = 12
        self.width = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.height),
            spaces.Discrete(self.width)
        ))
        self.moves = {
            0: (-1,0), # up
            1: (0,1), # right
            2: (1,0), # down
            3: (0,-1), # left
        }

        self.reset()

    def step(self, action):
        x,y = self.moves[action]
        self.S = random.choices(population=[(self.S[0]+x,self.S[1]+y),
                                (self.S[0]-x,self.S[1]+y),
                                (self.S[0]-x,self.S[1]-y),
                                (self.S[0]+x,self.S[1]-y)],weights=[0.9,0.1/3,0.1/3,0.1/3])[0]
        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        if self.S == (6,7):
            return self.S, 10, True, {}
        # puddle -1
        elif self.S[1]==3 and self.S[0] >= 2 and self.S[0] <= 8:
            return self.S,-1,False,{}
        elif self.S[1]==8 and self.S[0] >= 2 and self.S[0] <= 6:
            return self.S,-1,False,{}
        elif self.S[1]==7 and self.S[0] >= 6 and self.S[1] < 8:
            return self.S,-1,False,{}
        elif (self.S[0]==2 or self.S[0]==8) and self.S[1]>=4 and self.S[1]<=7:
            return self.S,-1,False,{}

        # puddle -2
        elif self.S[1]==4 and self.S[0] >= 3 and self.S[0] <= 7:
            return self.S,-2,False,{}
        elif self.S[1]==7 and self.S[0] >= 3 and self.S[0] <= 5:
            return self.S,-2,False,{}
        elif self.S[1]==6 and self.S[0] >= 5 and self.S[1] < 7:
            return self.S,-2,False,{}
        elif (self.S[0]==3 or self.S[0]==7) and self.S[1]>=4 and self.S[1]<=6:
            return self.S,-2,False,{}

        # puddle -3
        elif self.S[1]==5 and self.S[0] >= 4 and self.S[0] <= 6:
            return self.S,-3,False,{}
        elif self.S==(4,6):
            return self.S,-3,False,{}

        return self.S,-0.25,False,{}

    def reset(self):
        self.S = random.choices([5,6,10,11])[0],0
        return self.S