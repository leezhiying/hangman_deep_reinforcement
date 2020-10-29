#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:11:00 2019

@author: lizhiying
"""
import json
import requests
import random
import string
import secrets
import time
import re
import collections


class Environment(object):
    
    def __init__(self):
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location) 
        self.guessed = set()
        self.life = 15
        self.stop = False
        
        
    def step(self,action):
        
        reward = 0 
        
        
        if self.life == 0:
            reward = -1
            self.stop = True
        
        else:
            if self.state == self.truth:
                reward = 1 
                self.stop = True
            
            else:
            
                if action in self.truth:
                    reward = 0.01
                    self.state = self.update(action,self.truth,self.state)

                else:
                    self.life -= 1
                    reward = 0
        
        
        self.guessed.add(action)

        
        return {"state":self.state,"action":action,"reward":reward,"guessed_words":self.guessed,"life":self.life,'isterminated':self.stop}
    
    def reset(self):
        
        self.truth = list(random.sample(self.full_dictionary,1)[0])
        self.state = ['.']*len(self.truth)
        self.life = 15
        self.guessed = set()
        self.stop = False

        
        return self.state
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
    
    
    
    def update(self,guess,truth,state):
        for i in range(len(truth)):
            if truth[i] == guess:
                state[i] = guess
        
        return state
    
