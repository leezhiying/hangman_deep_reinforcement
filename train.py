# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:40:13 2019

@author: 90694
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Environment import Environment
      



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class DQN(nn.Module):
    
    def __init__(self,inputs,outputs):
        super(DQN, self).__init__()
         
        self.n1 = nn.Linear(inputs,100)
        self.n2 = nn.Linear(100, outputs)
        #self.n3 = nn.Linear(100,outputs)
        
    def forward(self,x):
        model = torch.nn.Sequential(
            self.n1,
            #nn.Dropout(p=0.6),
            nn.ReLU(),
            self.n2,
            nn.Softmax(dim=-1)
        )
        return model(x)


class HangmanAgent(object):
    

    
    
    
    def __init__(self):
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.full_letter  = set([chr(i) for i in range(97,97+26)])
        self.aeiou = set(['a','e','i','o','u'])
         
        self.memory =  []
        self.capacity = 10000
        self.position = 0 
        self.MAX_VOCABLENTH = 29
        self.steps_done = 0
        self.env = Environment()
        
        self.policy_net = DQN(30*27,26).to(self.device)
        self.target_net = DQN(30*27,26).to(self.device)
        
        #self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.target_net.eval()
        self.update_count = 0
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = 0.0001)
        
        self.train_reward_list = [] 
        
    def encode_state(self,observed,guessed):

# =============================================================================
#         We concatenate the word we have guessed to the observed state
#         Inputs: 
#         observed: a List contains what you observed from the environment 
#         guessed: a Set contains the letter you have 
#         Output: a numpy array shape 30*27 
#         
# =============================================================================

        word = [26 if i=='.' else ord(i)-97 for i in observed ]
        guessed_word_index = [ord(i)-97 for i in guessed]
        
        guessed_axis= np.zeros(27)
        for i in guessed_word_index:
            guessed_axis[i] = 1 
        
        
        state = np.zeros([30,27],dtype = np.float32)
        for i, j in enumerate(word):
           
            state[i, j] = 1
        
        state[-1] = guessed_axis
        
        return state
        

        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
        
    def push(self,*args):

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    
    
# =============================================================================
#   We adapt a strategy of epsilon greedy method with a decaying epsilon 
#    
# =============================================================================
    
    def select_action(self,state,guessed):
        EPS_START = 0.8
        EPS_END = 0.01
        EPS_DECAY = 100000
        
        sample = random.random()
        
        
        #self.eps_threshold = EPS_END + (EPS_START - EPS_END) *  math.exp(-1. * self.steps_done / EPS_DECAY)
        self.eps_threshold = 0
            
        self.steps_done += 1
        if sample > self.eps_threshold:
            state = torch.tensor(state).reshape(-1)
            #state = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #action = self.policy_net(state).max(1)[1].view(1, 1).item()
                
                action_mat = self.policy_net(state) 
                for item in guessed:
                    action_mat[ord(item)-97] = -99
              
                
               
       
                
                action = action_mat.argmax().item()
                
                
                action = chr(action + 97)
          
                return action
            
        else:
            action_space = self.full_letter - guessed 
            
            action_strategy = action_space & self.aeiou
            action = random.choice(list(action_space))
            
            if 'e' not in guessed:
                action = 'e'
            else:
                action = random.choice(list(action_space))
            
            return action

 
# =============================================================================
#     optimize model using methodology of DQN
# =============================================================================
    
    def optimize_model(self):
        
        BATCH_SIZE = 32
        GAMMA = 0.99
        TARGET_UPDATE = 100
  
        
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        #print(batch.action)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
    
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        

        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        state_action_values = torch.tensor(state_action_values,dtype= torch.double)
        
        #state_action_values = state_action_values.requires_grad_()
        state_action_values = state_action_values.clone().detach().requires_grad_(True)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        
        next_state_values = torch.tensor(next_state_values,dtype = torch.double)
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
      
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.data.clamp_(-1, 1)
        self.optimizer.step()
        
        if self.update_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.update_count += 1 
    
    
    

        
    
# =============================================================================
#     Start Training
# =============================================================================
    

    def train(self):
        reward_avg = 0 
        TARGET_UPDATE = 1000
        num_episodes = 100000
        for i_episode in range(1,num_episodes):
            #print("======================")
            #print(len(self.memory))    
            env = self.env
            env.reset()
           
            
            cum_reward = 0 
            
            state = self.encode_state(env.state,env.guessed)
             
            
            while True:
        
                action = self.select_action(state,env.guessed)
        
            
                result = env.step(action)
                next_state = result['state']
                reward = result['reward']
                guessed_word = result['guessed_words']
                done = result['isterminated']
             
                cum_reward += reward
              
                next_state = self.encode_state(next_state,guessed_word)
                
                reward = torch.tensor(np.asarray([reward],dtype = float), device=self.device)
                
                state_push = torch.tensor(state).unsqueeze(0).view(1,810)
                next_state_push = torch.tensor(next_state).unsqueeze(0).view(1,810)
                
                self.push(state_push, torch.tensor([[ord(action)-97]],dtype = torch.long,device=self.device),next_state_push, reward)

                
                state = next_state 
                self.optimize_model()
                if done:
                    break 
                
            #if cum_reward >= 1 and env.life > 0:
                #print('bingo')
            reward_avg += cum_reward
            

            if i_episode % 1000 == 0:
                reward_avg /= 1000
                print(i_episode,reward_avg,self.eps_threshold)
                self.train_reward_list.append(reward_avg)
                reward_avg = 0 
            
        
            
        return 
        
    

    
    
    
player = HangmanAgent()
player.train() 
plt.plot(player.train_reward_list)
    
    
    
    
    
    
    
    
    
    