import numpy as np
import random
import math
import os
import pickle

from model.model import model
import dataset

class AllCQAModel(model):
    def __init__(self, num_actions):
        super(AllCQAModel, self).__init__(num_actions)
        self.simu_flag = True
        self.successes = [0, 0]
        self.rewards = []
        self.first_turn = True
        self.temp_reward = 0.0
        self.action_choices = []
    
    def get_next_action(self, cur_state):
        #0 means None, 1 means IDENTIFICATION, 2 means ASK_CLARIFICATION, 
        #3 means CONFIRMATION, 4 means SELECT, 5 means REJECTION
        
        state = cur_state._cur_state
        action = -1
        print('received state:', state)
        if state.speaker != None:
            self.simu_flag = False
        else:
            self.simu_flag = True

        if state.speaker == None:
            action = 1
        
        elif state.speaker == 'L' and state.attachment_relation == 'NONE':
            action = 1

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            if np.argmax(state.distribution) == 0: #!= None and state.evaluation == 0:
                action = 3
            else:
                action = 1

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            if np.argmax(state.distribution) == 0:#state.evaluation != None and state.evaluation == 0:
                action = 3
            else:
                action = 5
        
        elif state.speaker == 'L' and state.attachment_relation == 'CONFIRMATION':
            action = 0
        
        elif state.speaker == 'L' and state.attachment_relation == 'REJECTION':
            action = 0

        elif state.speaker == 'S' and (state.attachment_relation == 'INFORMATION' or \
            state.attachment_relation == 'ASK_CLARIFICATION'):
            if self.first_turn:
                self.first_turn = False
                action = 2
            elif max(state.distribution) >= 0.95:
                self.first_turn = True
                action = 4
            else:
                self.first_turn = False
                action = 2

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            self.first_turn = True
            action = 4
        
        elif state.speaker == 'S' and state.attachment_relation == 'REJECTION':
            if max(state.distribution) >= 0.95:
                self.first_turn = True
                action = 4
            else:
                self.first_turn = False
                action = 2
        elif state.speaker == 'S' and state.attachment_relation == 'NONE':
            if self.first_turn:
                self.first_turn = False
                action = 2
            elif max(state.distribution) >= 0.95:
                self.first_turn = True
                action = 4
            else:
                self.first_turn = False
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
        elif action == 4:
            self.successes[1] += 1
            self.rewards.append(self.temp_reward)
            self.temp_reward = 0.0
        elif len(dialogue_history) == 1 and self.temp_reward != reward:
            self.successes[1] += 1
            self.temp_reward -= reward
            self.rewards.append(self.temp_reward)
            self.temp_reward = reward
    
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
