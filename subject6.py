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
    
    return reward

def CEM_ex(env, N = 100, rho = 0.1):

    env.reset()
    obs_dim = env.obs_dim()
    
    param = [random.uniform(-1, 1) for _ in range(obs_dim)]
    
    convergence = False
    count = 0

    tops = int(N*rho)

    top_rewards = []

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
        top_rewards.append(top_reward_mean)
        # top_rewards = np.append(top_rewards, np.array([top_rew]), axis = 0)
        
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
            
    return param, np.array(top_rewards)

class CPEnv_ex(CartPoleEnv):

    def __init__(self,
                 obs_noise = 0,
                 obs_drop = 0,
                 obs_complete = False):
        super().__init__()
        self.obs_noise = obs_noise
        self.obs_drop = obs_drop
        self.prev_obs_ex = None
        self.obs_complete = obs_complete

    def reset(self):
        self.prev_obs_ex = np.array(super().reset(), dtype = float)
        return self.prev_obs_ex

    def step(self, action):

        obs, terminate, reward = super().step(action = action)
        obs = np.array(obs, dtype = float)

        # noise term
        obs = obs * np.random.normal(1, self.obs_noise, obs.shape[0])

        # drop term
        drop_idx = np.random.choice([0,1],
                                    p = [self.obs_drop, 1-self.obs_drop],
                                    size = obs.shape[0])
        obs = obs * drop_idx

        # complete term
        if self.obs_complete:
            # complete dropped value(0) with previous obs
            obs[drop_idx == 0] = self.prev_obs_ex[drop_idx == 0]

        self.prev_obs_ex = obs

        return obs, terminate, reward 
        

def plot_witherror(ys, label):
    x = range(ys.shape[-1])
    y_mean = np.mean(ys, axis = 0)
    error = np.std(ys, axis = 0)
    plt.errorbar(x, y_mean, error, label = label)


# experiment for hyper-parameter(N, rho)
def exp1(env = CartPoleEnv()):

    Ns = [10, 50, 100]

    fig = plt.figure(figsize = (12,4))
    fig.subplots_adjust(bottom = 0.15)
    
    ax1 = fig.add_subplot(121)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('N = {}, rho = 0.1'.format(Ns))
    for n in Ns:
        rewards = np.array([train_ex(env = env,
                                     N = n,
                                     rho = 0.1)
                            for _ in range(10)])
        plot_witherror(rewards, label = 'N = {}'.format(n))
    plt.legend()

    rhos = [0.1, 0.3, 0.5]

    ax2 = fig.add_subplot(122)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('N = 100, rho = {}'.format(rhos))
    for rho in rhos:
        rewards = np.array([train_ex(env = env,
                                     N = 100,
                                     rho = rho)
                            for _ in range(10)])
        plot_witherror(rewards, label = 'rho = {}'.format(rho))
    plt.legend()

    with open('./tmp1.png', 'wb') as f:
        plt.savefig(f)

# experiment for partially missing obserbation
def exp2():

    drops = [0, 0.1, 0.3]

    fig = plt.figure(figsize = (12,4))
    fig.subplots_adjust(bottom = 0.15)
    
    ax1 = fig.add_subplot(121)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('observation drop (N = 100, rho = 0.1)')
    for drop in drops:
        rewards = np.array([train_ex(env = CPEnv_ex(obs_drop = drop),
                                     N = 100,
                                     rho = 0.1)
                            for _ in range(10)])
        plot_witherror(rewards, label = 'drop = {}'.format(drop))
    plt.legend()

    ax2 = fig.add_subplot(122)
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('observation reused (N = 100, rho = 0.1)')
    for drop in drops:
        rewards = np.array([train_ex(env = CPEnv_ex(obs_drop = drop,
                                                    obs_complete = True),
                                     N = 100,
                                     rho = 0.1)
                            for _ in range(10)])
        plot_witherror(rewards, label = 'drop = {}'.format(drop))
    plt.legend()

    with open('./tmp2.png', 'wb') as f:
        plt.savefig(f)

# experiment for noisy obserbation
def exp3():

    noises = [0, 0.2, 0.5]

    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('noisy observation')
    for noise in noises:
        rewards = np.array([train_ex(env = CPEnv_ex(obs_noise = noise),
                                     N = 100,
                                     rho = 0.1)
                            for _ in range(10)])
        plot_witherror(rewards, label = 'noise(std) = {}'.format(noise))
    plt.legend()

    with open('./tmp3.png', 'wb') as f:
        plt.savefig(f)
    

if __name__ == '__main__':

    # exp1(env = CPEnv_ex())
    exp2()
    exp3()

    CPEnv_ex().quit()

    
