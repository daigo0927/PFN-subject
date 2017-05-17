# coding:utf-8

import math
import random
import sys, os

from subject1 import EasyEnv

class LinearModel:

    def __init__(self, initial_param):
        self.param = initial_param
        self.act = None

    def action(self, obs):
        obs_value = [float(o) for o in obs]

        product = [p*o for p, o in zip(self.param, obs_value)]

        if sum(product)>0:
            self.act = 1
        else:
            self.act = -1

        return self.act

def test():
    
    env = EasyEnv()
    env.reset()

    obs_dim = env.obs_dim()
    
    init_param = [random.uniform(-1, 1) for _ in range(obs_dim)]
    print('obs_dim : {}, initial-param : {}'.format(obs_dim, init_param))
    model = LinearModel(initial_param = init_param)

    act = model.action(obs = [env.prev_obs])
    print('model parameter doesn\'t updated')
    
    while not env.terminate:
        print('obs , teminate , reward :{}'.format(env.step(action = act)))
        act = model.action(obs = [env.obs])
        print('next action {}'.format(act))
        

if __name__ == '__main__':
    test()
