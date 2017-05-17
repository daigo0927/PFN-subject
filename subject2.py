# coding:utf-8

import math
import random
import sys, os

class CartPoleEnv:

    def __init__(self):

        self.cleared = True
        self.reward = None
        self.prev_obs = None
        self.obs = None
        self.count = None
        self.terminate = False

    def reset(self):

        # send reset signal
        print('r')
        sys.stdout.flush()
        # receive initial observation
        feedback = input()
        self.prev_obs = feedback.split()

        self.count = 0

        return self.prev_obs

    def obs_dim(self):

        return len(self.prev_obs)-1

    def step(self, action):

        # send action, 1 or -1
        print('s {}'.format(action))
        sys.stdout.flush()

        # receive observation after the action
        feedback = input()
        self.obs = feedback.split()

        self.reward = 1

        self.prev_obs = self.obs

        self.count += 1
        if self.count == 500 or self.obs[0] == 'done':
            self.terminate = True

            # ternminate program
            print('q')
            sys.stdout.flush()

        return self.obs, self.terminate, self.reward

    


def test():
    env = CartPoleEnv()
    sys.stderr.write('env.reset() : {}\n'.format(env.reset()))
    sys.stderr.write('env.obs_dim() : {}\n'.format(env.obs_dim()))
    while not env.terminate:
        act = random.choice([-1, 1])
        sys.stderr.write('step{} env.step(action={}) : {}\n'\
                         .format(env.count, act, env.step(action = act)))


if __name__ == '__main__':
    test()
