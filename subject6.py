# coding:utf-8

import math
import random
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from operator import itemgetter

from subject1 import EasyEnv
from subject2 import CartPoleEnv
from subject3 import LinearModel

def train_ex(env, N = 100, rho = 0.1):
    env = env
    param, reward = CEM_ex(env = env, N = N, rho = rho)
    sys.stderr.write('final parameter {}\n'.format(param))
    sys.stderr.write('reward result shape : {}\n'.format(reward.shape))
    
    return param, reward

def CEM_ex(env, N = 100, rho = 0.1):

    env.reset()
    obs_dim = env.obs_dim()
    
    param = [random.uniform(-1, 1) for _ in range(obs_dim)]
    
    convergence = False
    count = 0

    tops = int(N*rho)

    top_rewards = np.empty((0, tops))

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
            

        top_record = sorted(Simu_record,
                            key = itemgetter(0),
                            reverse = True)[:tops]
        
        top_rew = [rec[0] for rec in top_record]
        top_reward_mean = sum(top_rew)/tops
        top_rewards = np.append(top_rewards, np.array([top_rew]), axis = 0)
        
        # sys.stderr.write('top {}'.format(top_record))
        env.reset()
        top_param = []
        for i in range(env.obs_dim()):
            param_i = [rec[1][i] for rec in top_record]
            top_param.append(sum(param_i)/tops)
            
        param = top_param
        
        count += 1
        
        sys.stderr.write('iterate {}, param {}, reward {}\n'\
                         .format(count, param, top_reward_mean))
        
        if count == 20:
            convergence = True
            
    return param, top_rewards

# experiment for effect of N, rho
def exp1():

    Ns = [10, 50, 100]

    fig = plt.figure(figsize = (12,4))
    
    ax1 = fig.add_subplot(121)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('N = {}, rho = 0.1'.format(Ns))
    for n in Ns:
        _, reward = train_ex(env = CartPoleEnv(),
                          N = n, rho = 0.1)
        plt.plot(np.mean(reward, axis = 1), label = 'N = {}'.format(n))
    plt.legend()

    rhos = [0.1, 0.3, 0.5]

    ax2 = fig.add_subplot(122)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('N = 100, rho = {}'.format(rhos))
    for rho in rhos:
        _, reward = train_ex(env = CartPoleEnv(),
                             N = 100, rho = rho)
        plt.plot(np.mean(reward, axis = 1), label = 'rho = {}'.format(rho))
    plt.legend()

    with open('./tmp.png', 'wb') as f:
        plt.savefig(f)

if __name__ == '__main__':

    exp1()

    print('q')
    sys.stdout.flush()
    # evaluate(param = param_trained)
    
