# coding:utf-8

import math
import random
import sys, os

from subject2 import CartPoleEnv
from subject3 import LinearModel
from subject4 import CrossEntropyMethod


def train():
    env = CartPoleEnv()
    param = CrossEntropyMethod(env = env)
    sys.stderr.write('final parameter {}'.format(param))
    # env.quit()
    return param

def evaluate(param):
    
    env = CartPoleEnv()
    env.reset()

    results = []
    for i in range(100):
        prev_obs = env.reset()
        agent = LinearModel(initial_param = param)

        # validation code
        if False: # i%2 == 0:
            agent = LinearModel(initial_param = [1,2,3,4])
            
        act = agent.action(obs = prev_obs)

        RewardSum = 0

        while not env.terminate:
            obs, _, reward = env.step(action = act)
            act = agent.action(obs = obs)
            RewardSum += reward

        if RewardSum >= 500:
            sys.stderr.write('{}-th episode successfully finised\n'.format(i))
            results.append(1)
        else:
            sys.stderr.write('{}-th episode failed\n'.format(i))
            results.append(0)

    score = sum(results)
    sys.stderr.write('trained parameter score : {}/100\n'.format(score))
    
    if score >= 95:
        param = [str(p) + '\n' for p in param]
        with open('./param.txt', 'w') as f:
            f.writelines(param)

    env.quit()
    

if __name__ == '__main__':
    
    param_trained = train()
    evaluate(param = param_trained)
    
