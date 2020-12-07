import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np
from collections import defaultdict

import random
import math

from model.model import model
import torch

  
matplotlib.style.use('ggplot')
steps_done = 0
EPS_START = 0.5

class QLearn(model):
    def __init__(self, num_actions, decay=2500, eps0=0.95, discount_factor=0.80, alpha=0.80):
        super(QLearn, self).__init__(num_actions)
        self.discount_factor = discount_factor
        self.alpha = alpha
        #call the parent here
        # self.num_actions = num_actions
        self.q_score = dict()
        self.eps_start = eps0
        self.eps_end = 0.00
        self.eps_decay = decay
        self.steps_done = 0
    
    def get_next_action(self, state):
        state = state.get_state_representation()
        sample = random.random()
        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) *\
            math.exp(-1. * self.steps_done/self.eps_decay)
        self.steps_done += 1

        if (self.steps_done > self.eps_decay or sample > eps_thresh) and state in self.q_score:
            print('Following Policy: ', self.steps_done)
            return np.argmax(self.q_score[state])
        else:
            return random.randint(0, self.num_actions - 1)

    def update_model(self, state, action, next_state, reward):
        # print('Adding the state to the representation: ', state)
        # print('Rewards: ', reward)
        if state not in self.q_score:
            self.q_score[state] = [0 for i in range(self.num_actions)]
        if next_state not in self.q_score:
            self.q_score[next_state] = [0 for i in range(self.num_actions)]
        
        new_value = reward + self.discount_factor * np.max(self.q_score[next_state]) - \
            self.q_score[state][action]
        self.q_score[state][action] += (self.alpha * new_value)
    
    def __str__(self):
        res = ''
        for item in self.q_score:
            res += str(item) + ' ' + str(self.q_score[item]) + '\n'
        return res
    
    def is_simulation(self):
        return False

    