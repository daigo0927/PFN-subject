# coding:utf-8

import math
import random
import sys, os

class EasyEnv:

    def __init__(self):

        self.cleared = True
        self.reward = None
        self.prev_obs = None
        self.obs = None
        self.count = None
        self.terminate = False

    def reset(self):

        # sampling initial obserbation from Uniform[-1, 1]
        self.prev_obs = [random.uniform(-1, 1)]

        self.count = 0

        self.terminate = False

        return self.prev_obs

    def obs_dim(self):

        return len(self.prev_obs)

    def step(self, action):

        # sampling current obserbation
        self.obs = [random.uniform(-1, 1)]

        self.reward = [action * prev for prev in self.prev_obs][0]

        self.prev_obs = self.obs

        self.count += 1
        if self.count == 10:
            self.terminate = True

        return self.obs, self.terminate, self.reward


def test():
    env = EasyEnv()
    print('env.reset() : {}'.format(env.reset()))
    print('env.obs_dim() : {}'.format(env.obs_dim()))
    for i in range(10):
        act = random.choice([-1, 1])
        print('step{} env.step(action={}) : {}'\
              .format(env.count, act, env.step(action = act)))


if __name__ == '__main__':
    test()
