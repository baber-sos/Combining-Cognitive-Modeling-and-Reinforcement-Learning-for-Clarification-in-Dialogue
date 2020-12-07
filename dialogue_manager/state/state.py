import numpy as np
import re
import torch
import copy
import os

# intent_file = open('/home/ubuntu/Dialogue_Research/color_in_context/system/dialogue_manager/state/intents')
intent_file = open(os.path.dirname(__file__) + '/intents')
INTENTS = [line.strip() for line in intent_file]

def get_clr_terms(logic_form):
    split_form = logic_form.split('AND')
    # split_form = [form_const.split('(')[1].split(')')[0].split(',') form_const in split_form]
    clr_terms = []
    for form_const in split_form:
        forms_args = form_const.split('(')[1].split(')')[0].split(',')
        cur_cterms = list(filter(lambda x : x.isdigit(), forms_args))
        if len(cur_cterms) >= 1:
            clr_terms += cur_cterms
    return [int(x) for x in clr_terms]

class list_node():
    def __init__(self, intent, form=None, nextn=None, prevn=None, \
        relation=None, speaker=None, evaluation=None, distribution=None, \
        tree=None, incomplete_parse=False):
        #value is the attachment relation with previous node.
        self.intent = intent
        self.form = form
        self.attachment_relation = relation
        self.speaker = speaker
        self.evaluation = evaluation
        self.distribution = distribution
        self.parse_tree = tree
        self.incomplete_parse = incomplete_parse
        self.clr_terms = get_clr_terms(self.form)

        self.next = nextn
        self.prev = prevn

    def __str__(self):
        return 'Intent:' + INTENTS[self.intent] + ' Intent Form: ' + str(self.form) + \
        ' | \nEvaluation: ' + str(self.evaluation) + ' | Attachment: ' + \
        str(self.attachment_relation) + ' | Speaker: ' + str(self.speaker) + ' | Distribution: ' + \
        str(self.distribution)

class state():
    def __init__(self):
        self.MAX_CONV_LEN = int(os.getenv('MAX_CONV_LEN'))
        self._cur_state = list_node(0, 'NONE()', distribution=np.array([0.33, 0.33, 0.33]), \
            relation='NONE', evaluation=-1)
        self.length = 0
        self._cur_representation = '(INIT(), None),(None, None);'
        if os.getenv('FEATURE_REP') == 'history':
            self._vector = torch.zeros( (1, (((self.MAX_CONV_LEN + 1) * len(INTENTS)) + 3 + 1)) )
        elif os.getenv('FEATURE_REP') == 'actions':
            self._vector = torch.zeros((1, 3 + (len(INTENTS)**2) + 1 + 1))
    
    def is_init(self):
        return self._cur_state.form == 'INIT()'
    
    def update(self, intent, new_state, evaluation, shuffle_order, tree=None, incomplete_parse=False):
        state_update = new_state[0]
        speaker = new_state[1]

        to_add = list_node(intent=intent, form=state_update, relation=evaluation[1], \
            speaker=speaker, evaluation=np.argmax(evaluation[0]), distribution=evaluation[0], \
            tree=tree, incomplete_parse=incomplete_parse)
        
        # print('########################')
        # print('This tree was incomplete:', incomplete_parse)
        # print('########################')

        to_add.prev = self._cur_state
        self._cur_state.next = to_add
        self._cur_state = to_add

        #update representations
        self._cur_representation += str(new_state) + ',' + str(evaluation) + ';'
        self.update_vector_representation(shuffle_order)

        self.length += 1


    def get_state_representation(self):
        return self._cur_representation
    
    def update_vector_representation(self, order):
        # print('**************State Right Now:\n', type(self._cur_state.evaluation), '**************')
            # print('********')
            # print(self._cur_state)
            # print('********')
        # dist = [self._cur_state.distribution[x] for x in order]
            # print(dist, self._cur_state.distribution)
            # print('********')
        cur_dist = [x for x in self._cur_state.distribution]
        cur_dist.sort()
        self._vector[0][0:3] = torch.tensor(cur_dist).to(os.getenv('DEVICE'))
        
        # print('Length of the vector:', len(self._vector))
        # print(':', len(self._vector))
        if os.getenv('FEATURE_REP') == 'history':
            self._vector[0][(self.length * len(INTENTS)) + self._cur_state.intent + 3] = 1
        elif os.getenv('FEATURE_REP') == 'actions':
            # self._vector[0][self._cur_state.intent + 3] = 1
            self._vector[0][(self._cur_state.intent * len(INTENTS)) + self._cur_state.intent + 3] = 1
            self._vector[0][-2] = (self.length + 1)/(int(os.getenv('MAX_CONV_LEN')) + 1)

        if self._cur_state.incomplete_parse:
            self._vector[0][-1] = 1
        else:
            self._vector[0][-1] = 0
    
    def get_vector_representation(self):
        return self._vector.clone().detach()
    
    def __len__(self):
        return self.length
    
    def __str__(self):
        return 'Intent Form: ' + str(self._cur_state.form) + ' Evaluation: ' + \
            str(self._cur_state.evaluation) + ' Attachment: ' + \
            str(self._cur_state.attachment_relation)
    
