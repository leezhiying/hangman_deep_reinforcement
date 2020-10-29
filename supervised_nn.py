# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:37:45 2019

@author: zhiying.li
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



 

#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)






class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(240, 50)
        self.fc2 = nn.Linear(50, 26)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 240)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
 
        self.fc1 = nn.Linear(30*27, 256)
        self.fc2 = nn.Linear(256, 26)

    def forward(self, x):
 
        x = F.relu(self.fc1(x))
 
        x = self.fc2(x)
        return F.softmax(x)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, middle_layer_size,output_size,device):
        super(RNN, self).__init__()
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
 
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2n = nn.Linear(input_size + hidden_size, middle_layer_size)
        self.i2o = nn.Linear(middle_layer_size, output_size)
        
        self.dropout = nn.Dropout(0.1)
 
    def forward(self, input, hidden):
 
        combined = torch.cat((input, hidden), 1).to(device)
 
        hidden = self.i2h(combined) 
        middle = self.i2n(combined)
        
        output = self.i2o(middle)
        
        #output = self.dropout(output)
        output = F.softmax(output)
        return output, hidden 
 
    def init_hidden(self):
 
        return torch.zeros(1, self.hidden_size).to(device)
    
    
















def encode_state(observed,guessed):

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



def encode_answer(hidden):
    response = np.zeros(26, dtype=np.float32)
    for item in hidden:
        loc = ord(item) - 97
        response[loc] = 1
        
    response /= response.sum()
    
    return(response)
  














def data_generate():

    X = [ ] 
    y = [ ]
    
    NUM_SAMPLE = 100000
    for i in range(NUM_SAMPLE):
        if i % 1000 == 0:
            print(i)
        env = Environment()
        env.reset()
        word = env.truth
        state = env.state
        while True:
            #print(state)
            state = encode_state(state,env.guessed)
            #X.append(torch.tensor(state))
            X.append(state)
            
            hidden = [item for item in word if item not in env.guessed ]
            c = Counter(hidden)
            action = c.most_common()[0][0]
            
            hidden = set(hidden)
            answer = encode_answer(hidden)
            
            y.append(answer)
 
            result = env.step(action)
            
             
            #y.append( torch.tensor([ord(action)-97],dtype = torch.long,device=device))
            
            
            next_state = result['state']
    
            reward = result['reward']
            done = result['isterminated']
            state = next_state 
            
            if state == word:
                #print('bingo')
                break
        


 
     
        
    return X,y




if __name__ == '__main__':
    '''
    X,y = data_generate()
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    torch.save(X_tensor,'X_tensor.pt')
    torch.save(y_tensor,'y_tensor.pt')
    '''
    
    
    # =============================================================================
    # training !!!!
    # =============================================================================
    
    X = torch.load('X_tensor.pt')
    y = torch.load('y_tensor.pt')
    
    
    
    #net = SimpleCNN().to(device)
    #net = MLP().to(device)
    net = RNN(input_size = 27,hidden_size = 27, middle_layer_size = 64,output_size = 26,device = device).to(device)
    
    #net = nn.RNN(input_size = 27,hidden_size = 27, num_layers = 26).to(device)
    
    
    training_start_time = time.time()
    BATCH_SIZE = 32
    
    NUM_EPOCHS = 1
    
    loss = F.smooth_l1_loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    
    def generate_batch(X,y,batch):
        
        #s = np.random.choice(np.arange(len(X)),BATCH_SIZE)
        
        #X_batch = [np.array(X[i]) for i in s]
        #y_batch = [np.array(y[i]) for i in s]
        
        
        X_batch = [np.array(X[batch])]
        y_batch = [np.array(y[batch])]
        
        return np.array(X_batch),np.array(y_batch)
    
    
    train_losses  = []
    
    running_loss = 0.0
    
    for i_epoch in range(0,NUM_EPOCHS):   
        
        for batch in range(len(X)):
        
            start_time = time.time()
            
            
            X_batch,y_batch = generate_batch(X,y,batch)
            hidden = X_batch[0][-1]
            X_batch = Variable(torch.Tensor(X_batch)) 
            y_batch = Variable(torch.Tensor(y_batch)).to(device)
     
            
            optimizer.zero_grad()
            #X_batch = X_batch.view(32,1,30,27).to(device) # This is used for CNN
            #X_batch = X_batch.view(32,30*27).to(device) # THis is used for Simple MLP
            
            
            X_batch = X_batch.view(30,1,27).to(device)
            hidden = Variable(torch.Tensor(hidden)).to(device)
            hidden = hidden.view(1,27)
           
            for i in range(X_batch.size()[0]):
                outputs, hidden = net(X_batch[i], hidden)
     
            
            loss_size = loss(outputs, y_batch)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.item()
            
            if batch  % 1000 == 0:
              print('Train Epoch:',i_epoch, 'batch:',batch, 'loss:',running_loss/1000,'time',time.time() - training_start_time )
              torch.save(net,"model.pkl")
              train_losses.append(loss_size.item())
              running_loss =0
              
    plt.plot(train_losses)
    torch.save(net,"model.pkl")




