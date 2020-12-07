
import random
import math

from model.model import model

class QAModel(model):
    def __init__(self, num_actions):
        self.num_actions = num_actions
    
    def get_next_action_1(self, cur_state):
        #0 means IDENTIFICATION, 1 means ASK_CLARIFICATION, 2 means CONFIRMATION
        # print('received state:', state)
        state = cur_state._cur_state

        if state.speaker == None and state.attachment_relation == None:
            return 0

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 2

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.choice([0, 2])
        
        elif state.speaker == 'L' and state.attachment_relation == 'CONFIRMATION':
            return 2

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            return random.randint(1,2)

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            return random.randint(1,2)
        else:
            return 1


def get_next_action_2(self, cur_state):
        #0 means IDENTIFICATION, 1 means ASK_CLARIFICATION, 2 means CONFIRMATION
        # print('received state:', state)
        state = cur_state._cur_state

        if state.speaker == None and state.attachment_relation == None:
            return 0

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 2

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.choice([0, 2])
    
        else:
            return 2

def get_next_action_3(self, cur_state):
        #0 means IDENTIFICATION, 1 means ASK_CLARIFICATION, 2 means CONFIRMATION
        # print('received state:', state)
        state = cur_state._cur_state

        if state.speaker == None and state.attachment_relation == None:
            return 0

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 2

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.choice([0, 2])

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            return random.randint(1,2)

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            return random.randint(1,2)
        else:
            return 1


def get_next_action_4(self, cur_state):
        #0 means IDENTIFICATION, 1 means ASK_CLARIFICATION, 2 means CONFIRMATION
        # print('received state:', state)
        state = cur_state._cur_state

        if state.speaker == None and state.attachment_relation == None:
            return 0

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 2

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.choice([0, 2])

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            return random.randint(1,2)

        elif state.speaker == 'S' and state.attachment_relation == 'CONFIRMATION':
            return random.randint(1,2)
        else:
            return 1

def get_next_action_5(self, cur_state):
        #0 means IDENTIFICATION, 1 means ASK_CLARIFICATION, 2 means CONFIRMATION
        # print('received state:', state)
        state = cur_state._cur_state

        if state.speaker == None and state.attachment_relation == None:
            return 0

        elif state.speaker == 'L' and state.attachment_relation == 'INFORMATION':
            return 2

        elif state.speaker == 'L' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.choice([0, 2])

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            return random.randint(1,2)

        elif state.speaker == 'S' and state.attachment_relation == 'ASK_CLARIFICATION':
            return random.randint(1,2)

        elif state.speaker == 'S' and state.attachment_relation == 'INFORMATION':
            return random.randint(1,2)

        else:
            return 1


        
    def update_model(self, state, action, next_state, reward):
        pass
