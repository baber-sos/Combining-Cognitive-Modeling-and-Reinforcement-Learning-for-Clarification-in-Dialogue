
import random
import math

from model.model import model
import json
import os
import numpy as np

import torch
from torch.distributions.categorical import Categorical as cat_dist

class SQAModel(model):
    def __init__(self, num_actions, cic=None, ix=-1):
        item = cic._df.iloc[cic[ix]['row_index']]
        self.dialog = json.loads(item.loc['full_text'])
        super(SQAModel, self).__init__(num_actions)
        self.flag = True
    
    def get_next_action(self, cur_state):

        #0 means None, 1 means IDENTIFICATION, 2 means ASK_CLARIFICATION, 
        #3 means CONFIRMATION, 4 means SELECT, 5 means REJECTION
        
        state = cur_state._cur_state
        # print(state)
        if state.speaker == None and state.attachment_relation == 'NONE':
            role = ''
            count = 0
            current_description = ''
            while self.flag and role != 'speaker' and count < len(self.dialog):
                current_description = self.dialog[count]['text'].strip()
                role = self.dialog[count]['role']
                count += 1
            if count > 0:
                self.flag = False
            # print('first utterance:', current_description, 'role:', role)
            if role == 'speaker':
                return (current_description, 'IDENTIFY')
            return 1

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 0

        elif state.speaker == 'L' and state.intent == 2:
            if int(cat_dist(torch.tensor(state.distribution)).sample()) == 0:
               if random.random() <= float(os.getenv('MIS_ID1')):
                   return 5
               else:
                   return 3
            # if np.argmax(state.distribution) == 0:
                #return 3
            else:
                if random.random() <= float(os.getenv('PARSE_MIS')):
                    return 0
                prob_thresh = random.random()
                #if prob_thresh <= 0.45:
                #return 5
                #else:
                return 1
        elif state.speaker == 'L' and state.intent == 6:
            sorted_ind = np.argsort(state.distribution)[::-1]
            if 0 in sorted_ind[:2]:
                return 1
            elif random.random() <= float(os.getenv('PARSE_MIS')):
                return 0
            return 1
        elif state.speaker == 'L' and state.intent == 7:
            return 1
        elif state.speaker == 'L' and state.attachment_relation == 'CONFIRMATION':
            return 0
        
        elif state.speaker == 'L' and state.attachment_relation == 'REJECTION':
            return 0
        
        elif state.speaker == 'L' and state.attachment_relation == 'NONE':
            return 0

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            # print('I AM HERE!', state)
            return random.choice([2,4])

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            if state.evaluation == 'YES':
                return random.choice([4, 5])
            else:
                return 2
        else:
            return 2
        
    
    def is_simulation(self):
        return True
