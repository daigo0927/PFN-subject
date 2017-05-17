# coding:utf-8

import math
import random
import sys, os

from operator import itemgetter

from subject1 import EasyEnv
from subject3 import LinearModel

def CrossEntropyMethod(env, N = 100, rho = 0.1):
    
    env.reset()
    obs_dim = env.obs_dim()

    param = [random.uniform(-1, 1) for _ in range(obs_dim)]

    convergence = False
    count = 0

    while not convergence:
        Simu_record = []
        
        for i in range(N):

            RewardSum = 0
            param_noised = [random.gauss(0, 1) + par for par in param]
            agent = LinearModel(initial_param = param_noised)

            prev_obs = env.reset()

            act = agent.action(obs = prev_obs)
            
            while not env.terminate:
                obs, _, reward = env.step(action = act)
                RewardSum += reward
                act = agent.action(obs = obs)

            Simu_record.append((RewardSum, param_noised))

        tops = int(N*rho)
        top_record = sorted(Simu_record,
                            key = itemgetter(0),
                            reverse = True)[:tops]
        
        top_rew = [rec[0] for rec in top_record]
        top_reward = sum(top_rew)/tops

        # sys.stderr.write('top {}'.format(top_record))
        env.reset()
        top_param = []
        for i in range(env.obs_dim()):
            param_i = [rec[1][i] for rec in top_record]
            top_param.append(sum(param_i)/tops)

        param = top_param
        
        count += 1
        
        sys.stderr.write('iterate {}, param {}, reward {}\n'\
                         .format(count, param, top_reward))

        if count == 100:
            convergence = True

    return param
            
        
def test():

    env = EasyEnv()
    param = CrossEntropyMethod(env = env)

    sys.stderr.write('final parameter : {}\n'.format(param))
            

if __name__ == '__main__':
    test()
