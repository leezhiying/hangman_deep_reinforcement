# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:36:05 2019

@author: 90694
"""

import time 
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import Counter
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from Environment import Environment
 
from supervised_nn import SimpleCNN,encode_state,RNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load("model.pkl").to(device)
torch.manual_seed(595)
np.random.seed(5951)
random.seed(5951)
 

NUM_EPISODES = 1000
cum_reward = 0

success = 0

for i in range(NUM_EPISODES):
    print("="*20)
    env = Environment()
    env.reset()    
    state = env.state

    done = False 
    reward = 0
    while True:
        print(state)
    #print(state)
        if done:
            cum_reward += reward
            if env.life > 0:
                print('bingo')
                success += 1
            else:
                print('no')
            
            break
        
        state = encode_state(state,env.guessed)
       
        state = torch.Tensor(state).view(30,1,27).to(device)
        hidden = state[-1].view(1,27).to(device)
        #state = state.view(1,30,27)
        for i in range(state.size()[0]):
            action_mat , hidden = net(state[i],hidden)
        
        action_mat = action_mat[0]
        for item in env.guessed:
            action_mat[ord(item)-97] = -99
        
        action = action_mat.argmax().item()
        action = chr(action + 97)
 
         
        result = env.step(action)
        
        
        
        next_state = result['state']
        
        reward = result['reward']
        done = result['isterminated']
        state = next_state 

print("success_rate",success/NUM_EPISODES)     
