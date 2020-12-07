import numpy as np
import random
import math
import os
import pickle

from model.model import model
import dataset
from torch.distributions.categorical import Categorical as cat_dist

class QAC3Model(model):
    def __init__(self, num_actions):
        super(QAC3Model, self).__init__(num_actions)
        self.simu_flag = True
        self.successes = [0, 0]
        self.rewards = []
        self.first_turn = True
        self.temp_reward = 0.0
        self.action_choices = []
    
    def get_next_action(self, cur_state):
        #0 means None, 1 means IDENTIFICATION, 2 means ASK_CLARIFICATION with one CP, 
        #3 means CONFIRMATION, 4 means SELECT, 5 means REJECTION,
        #6 means Clarification as I don't know, 7 means Clarification with 2 CP's,
        #8 means Clarification with 3 CP's
        
        state = cur_state._cur_state
        action = -1
        # print('received state:', state)
        if state.speaker != None:
            self.simu_flag = False
        else:
            self.simu_flag = True

        if state.speaker == None:
            action = 1
        
        elif state.speaker == 'L' and state.attachment_relation == 'NONE':
            action = 1

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            # if int(cat_dist(torch.tensor(state.distribution)).sample()) == 0:
            # # if np.argmax(state.distribution) == 0: #!= None and state.evaluation == 0:
            #     action = 3
            # else:
            action = 1

        elif state.speaker == 'L' and state.intent == 2:
            # if int(cat_dist(torch.tensor(state.distribution)).sample()) == 0:
            # # if np.argmax(state.distribution) == 0:#state.evaluation != None and state.evaluation == 0:
            #     action = 3
            # else:
            prob_thresh = random.random()
            if prob_thresh <= 0.45:
                action = 5
            else:
                action = 1
        elif state.speaker == 'L' and state.intent == 6:
            action = 1
        elif state.speaker == 'L' and state.intent == 7:
            action = 1
        elif state.speaker == 'L' and state.attachment_relation == 'CONFIRMATION':
            action = 0
        
        elif state.speaker == 'L' and state.attachment_relation == 'REJECTION':
            action = 0

        elif state.speaker == 'S' and (state.attachment_relation == 'INFORMATION' or \
            state.attachment_relation == 'ASK_CLARIFICATION'):
            if max(state.distribution) >= 0.95:
                action = 4
            else:
                action = 2

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            if max(state.distribution) >= 0.95:
                action = 4
            else:
                action = 2
        elif state.speaker == 'S' and state.attachment_relation == 'REJECTION':
            if max(state.distribution) >= 0.95:
                action = 4
            else:
                action = 2
        elif state.speaker == 'S' and state.attachment_relation == 'NONE':
            if max(state.distribution) >= 0.95:
                action = 4
            else:
                action = 2
        
        self.action_choices.append((tuple(state.distribution), action))
        return action
    
    def is_simulation(self):
        return True
    
    def update_model(self, dialogue_history, state, action, next_state, reward):
        self.temp_reward += reward
        if action == 4 and reward >= 0.3:
            self.successes[0] += 1
            self.rewards.append(self.temp_reward)
            self.temp_reward = 0.0
        elif action == 4 or len(dialogue_history) == int(os.getenv('MAX_CONV_LEN')) + 1:
            self.successes[1] += 1
            self.rewards.append(self.temp_reward)
            self.temp_reward = 0.0
        #elif len(dialogue_history) == 2 and self.temp_reward != reward:
        #    # print('Adding to ')
        #    self.successes[1] += 1
        #    self.temp_reward -= reward
        #    self.rewards.append(self.temp_reward)
        #    self.temp_reward = reward
        # elif len(dialogue_history) == 1 :
        #     self.successes[1] += 1
    
    def save_model(self):
        dir_path = os.path.dirname(os.path.dirname(dataset.__file__))
        file_path = dir_path + '/model/training_parameters/test_successes' + \
            'ruleBased' + os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + '.pkl'
        pickle.dump([self.successes], open(file_path, 'wb'))
        file_path = dir_path + '/model/training_parameters/test_reward_track' + \
            'ruleBased' + os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + '.pkl'
        pickle.dump([self.rewards], open(file_path, 'wb'))
        file_path = dir_path + '/model/training_parameters/test_action_choices' + \
            'ruleBased' + os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + '.pkl'
        pickle.dump([self.action_choices], open(file_path, 'wb'))
