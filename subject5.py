# coding:utf-8

import math
import random
import sys, os

from subject2 import CartPoleEnv
from subject4 import CrossEntropyMethod


def train():
    env = CartPoleEnv()
    param = CrossEntropyMethod(env = env)
    sys.stderr.write('final parameter {}'.format(param))
    env.quit()

if __name__ == '__main__':
    train()
