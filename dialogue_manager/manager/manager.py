import sys

from state.state import INTENTS
from state.state import state

from manager.generate_intent_argument import *
from manager.evaluate import evaluate

from magis.models.model_b import ModelBWithRGC
from magis.settings import REPO_ROOT

from chart_parser.astar_parse import chart_parse
from chart_parser.logic import get_logical_form
from chart_parser.util import preprocess
from chart_parser.util import get_correct_sentence

# from generation.gen-rules import generation, generation_map

import os
import torch
import random

class manager(object):
    def __init__(self, colors, speaker_model, listener_model, \
        grammar, intent_map, cic=None, mode='fft', topk=10, first_turn='S', batch_dict=None):

        # print('Constructor of manager')
        self._speaker_rules = list()
        self._listener_rules = list()
        self._state_rep = state()
        # self._model = ModelBWithRGC.from_pretrained(os.path.join(REPO_ROOT, "models", "LUX2B"))
        # if self.batch_dict == None:
        #     self._model = cic._model#get_model_interface(cic)
        # else:

        #this contains all the probabilities for the color patches and 
        #does not contain the color model object like it did before.
        self._model = (batch_dict, cic._model[1])
        self._cic = None

        if cic == None:  
            self._cic = make_or_load_cic()
        else:
            self._cic = cic

        self.grammar, self.terminals, self.non_terminals, self.first_set = grammar
        self.irmap, self.rimap = intent_map
        # print('RIMap:', )
        # print('##############################')
        # print('IRMap:', irmap)
        # print('##############################')
        self._color_vocab = self._cic._color_vocab
        self._colors = {'S':[None, None, None],
            'L':[None, None, None]}
        self._topk = topk
        
        self._shuffle_order = {'S' : self.get_shuffle_order(colors, 'S'), 
            'L' : self.get_shuffle_order(colors, 'L')}
        # print('**********************')
        # print('This is the shuffle order:', self._shuffle_order)
        # print('**********************')
        if mode == 'fft':
            for turn in ['S', 'L']:
                self._colors[turn] = [torch.tensor(x).view(1, -1) for x in colors[turn]]
            for turn in ['S_RGB', 'L_RGB']:
                self._colors[turn] = [np.array(x)/255 for x in colors[turn]]
            # print('$$$$&&&&&&&&&&Tensor Shape:', self._colors[turn])
        elif mode == 'rgb':
            pass

        self._this_turn = first_turn
        self._turn_count = 0
        self._done = False

        #I get the model from the test to decouple it
        self._conversation_model = {'S' : speaker_model,
            'L' : listener_model}
    
    def update_turn(self, cur_turn=None):
        if cur_turn != None:
            self._this_turn = cur_turn
        if self._this_turn == 'S':
            self._this_turn = 'L'
        else:
            self._this_turn = 'S'
        self._turn_count += 1

    def tokenize(self, string):
        # print('This is the received string:', string)
        symbol_list = ['.', ',', '?', '/', '\\', '?', '-', ';', '!']
        utterance = string.lower()
        for s in symbol_list:
            utterance = (' ' + s + ' ').join(utterance.split(s)).strip()
        return utterance.split()
    
    def get_logic_from_sentence(self, text):
        new_move = []
        text = get_correct_sentence(self.tokenize(text), self.terminals, self._cic.spellcheck_map)
        parse_tree = chart_parse(preprocess(text, self._color_vocab), \
            self.grammar, self.terminals, self.non_terminals, self.first_set, '<P>')
        to_check = ''
        incomplete_parse = False
        for i, partial in enumerate(parse_tree):
            if i >= 1:
                incomplete_parse = True
            if partial:
                partial.print_tree()
                str_tree = str(partial.to_str())
                logic_form = get_logical_form(self._color_vocab, self.rimap, self.irmap,\
                    partial, 'F')[0]
                new_move.append((logic_form, str_tree))
                to_check += (logic_form + '\n')
        if new_move == []:
            return [('NONE()', str(('None',)))], 'NONE', True
        elif 'REJECTION' in to_check:
            return new_move, 'REJECTION', incomplete_parse
        elif 'CONFIRMATION' in to_check:
            return new_move, 'CONFIRMATION', incomplete_parse
        elif 'IDENTIFY' in to_check:
            return new_move, 'IDENTIFY', incomplete_parse
        
    def state_to_str(self, state):
        to_ret = (state.parse_tree if state.parse_tree else 'NONE') + '\n' + state.form + '\n';
        distribution = state.distribution
        if state.speaker:
            distribution = [state.distribution[i] for i in self._shuffle_order[state.speaker]]
        to_ret += str(distribution)
        return to_ret
    
    def check_validity(self, prev_state, cur_action):
        # print('$$$$$$$$$$$$$$$4This is the previous state:', prev_state)
        # print('THIS IS THE CURRENT ACTION:', cur_action)
        if prev_state.speaker == 'L' or prev_state.speaker == None:
            return True
        
        if cur_action in [2,4,6,7]:
            return True
        elif self._this_turn == 'L' and prev_state.attachment_relation == 'NONE' and cur_action == 0:
            return True
        else:
            return False

    def get_next_move(self, utterance=None):
        if self._done:
            # print('THIS IS DONE FOR SOME REASON')
            return '', None, self._done
        if self._turn_count > int(os.getenv('MAX_CONV_LEN')):
            self._done = True
            return '', 'MAX', True
            
        if utterance != None:
            temp_move, temp_intent, incomplete_parse = self.get_logic_from_sentence(utterance['text'])

            if temp_intent == 'NONE':
                return '', 'NONE()', False

            temp_ix = [x == temp_intent for x in INTENTS].index(True)
            # print('#############The list of obtained moves:', temp_move)
            for move in temp_move:
                temp_evaluation = evaluate(self._model, self._cic.adj_model, \
                    self._colors, ('S' if utterance['name']=='L' else 'L'),\
                    self._color_vocab, self._state_rep, move[0], self._shuffle_order)

                self._state_rep.update(temp_ix, (move[0], utterance['name']), \
                    temp_evaluation[1:], self._shuffle_order[utterance['name']], \
                    tree=move[1], incomplete_parse=incomplete_parse)
            # print(utterance['name'] + ':', temp_move, '| Evaluation:', temp_evaluation)
            # print('####################')
            self.update_turn(utterance['name'])

        mv_ix = None
        move_intent = ''
        new_move = ''
        next_action = ''
        tree = None
        if self._conversation_model[self._this_turn] == None:
            mv_ix = random.randint(0, len(INTENTS) - 1)
        else:
            next_action = self._conversation_model[self._this_turn].get_next_action(\
                self._state_rep)
            if type(next_action) == tuple:
                new_move, move_intent, incomplete_parse = self.get_logic_from_sentence(next_action[0])
                move_intent = next_action[1]
                mv_ix = [x == move_intent for x in INTENTS].index(True)
                # for move in new_move:
                tree = new_move[0][1]
                new_move = new_move[0][0]
                
            else:
                incomplete_parse = False
                if next_action == 0 and self._this_turn == 'S':
                    incomplete_parse = True
                move_intent = INTENTS[next_action]
                mv_ix = next_action
                new_move = self.generate_move(move_intent)

        # prev_evaluation = np.array([0.333, 0.333, 0.333])
        # for move, tree in new_move:
        new_evaluation = evaluate(self._model, self._cic.adj_model, \
            self._colors, ('S' if self._this_turn=='L' else 'L'), \
            self._color_vocab, self._state_rep, new_move, self._shuffle_order)
            # temp_eval = new_evaluation[1] * prev_evaluation
            # temp_eval = temp_eval/sum(temp_eval)
            # new_evaluation = (new_evaluation[0], temp_eval, new_evaluation[2])
        
        # print(self._this_turn + ':', new_move, '| Evaluation:', new_evaluation)
        # print('####################')
        # if self._this_turn == 'L':
            # reward = max(-1.0, new_evaluation[0] * (2**(self._turn_count//2)))
        if mv_ix == 2:
            reward = eval(os.getenv('RA'))
        elif mv_ix == 6:
            reward = eval(os.getenv('RB')) 
        elif mv_ix == 7:
            reward = eval(os.getenv('RC')) 
        else:
            reward = new_evaluation[0]
         
        
        cur_state_rep = self._state_rep.get_vector_representation()
        
        #get the previous state in string form here
        decision_info = None
        if os.getenv('VERBOSE') == 'True':
            decision_info = self.state_to_str(self._state_rep._cur_state)

        not_valid = False
        if not self.check_validity(self._state_rep._cur_state, mv_ix):
            # print('Wrong Move:', move_intent)
            not_valid = True
        
        if not_valid or self._turn_count >= int(os.getenv('MAX_CONV_LEN')):
            self._state_rep.update(len(INTENTS) - 1, ('NONE()', self._this_turn), \
                (np.array([0, 0, 0]), 'NONE'), self._shuffle_order[self._this_turn], \
                tree=tree, incomplete_parse=False)
        else:
            self._state_rep.update(mv_ix, (new_move, self._this_turn), \
                new_evaluation[1:], self._shuffle_order[self._this_turn], \
                tree=tree, incomplete_parse=incomplete_parse)

        if not_valid == False and self._this_turn == 'L' and os.getenv('TRIAL_MODE') == 'False':
            next_action = self._conversation_model['S'].get_next_action(self._state_rep)
            if next_action == 3 or next_action == 5:
                reward -= float(os.getenv('OLD_TERM'))
            elif next_action == 1 and self._state_rep._cur_state.intent == 6:
                dist = self._state_rep._cur_state.distribution
                arg_sort = dist.argsort()[::-1]
                reward -= (float(os.getenv('OLD_TERM')) if 0 in arg_sort[:2] else float(os.getenv('NEW_TERM')))
            elif next_action == 1 and self._state_rep._cur_state.intent == 2:
                reward -= float(os.getenv('NEW_TERM'))
            elif next_action == 1 and self._state_rep._cur_state.intent == 7:
                reward -= float(os.getenv('OLD_TERM'))

        next_state_rep = self._state_rep.get_vector_representation()
        action = mv_ix
        
        # new_move = [agent_mind, new_move]
        # print('THIS IS MOVE INTENT:', move_intent, 'NOT VALID:', not_valid)
        if self._this_turn == 'L' and (move_intent == 'SELECT' or \
            move_intent == 'CONFIRM_SELECT'):
            self._done = True
            reward = new_evaluation[0]
            self._conversation_model[self._this_turn].update_model(self._state_rep, cur_state_rep, action, \
                next_state_rep, reward)

            return decision_info, new_move, self._done
        elif not_valid == True or self._turn_count >= int(os.getenv('MAX_CONV_LEN')):
            self._done = True
            reward = float(os.getenv('RINV'))
            self._conversation_model[self._this_turn].update_model(self._state_rep, cur_state_rep, action, \
                next_state_rep, reward)

            return decision_info, new_move, self._done
        # if reward == 0:
            # print('WHAT IS THIS:', state._state_rep)
        # print(self._this_turn, reward, state)
        for k in [self._this_turn]:
            if not self._conversation_model[k]:
                continue
            self._conversation_model[k].update_model(self._state_rep, cur_state_rep, action, \
                next_state_rep, reward)
            # self._conversation_model['L'].update_model(self._state_rep, cur_state_rep, action, \
            #     next_state_rep, reward)

        if 'NONE' not in new_move or utterance == None:
            self.update_turn()
        else:
            new_move = utterance['text']

        return decision_info, new_move, self._done

    def generate_move(self, intent):
        #this function will take the generated state and intent as an argument
        #and generate the next move
        # print('I am goign to call this person: ', intent)
        # print('evaluation:', eval( '(self._this_turn, 
        #self._conversation_model[self._this_turn].is_simulation())' ))
        # print(intent)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('THIS IS THE TURN', self._this_turn)
        # print(intent, self._colors[self._this_turn])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@')
        return eval('gen_' + intent + \
            '(self._model, self._colors[self._this_turn], self._state_rep._cur_state,' + \
            '(self._this_turn, self._conversation_model[self._this_turn].is_simulation()), self._color_vocab)')
    
    def get_shuffle_order(self, colors, speaker):
        reference_colors = colors[speaker]
        other_speaker = 'S' if speaker == 'L' else 'L'
        # print(speaker, other_speaker)
        result = []
        for j in reference_colors:
            for i, clr in enumerate(colors[other_speaker]):
                # print(j, clr)
                if (clr == j).all():
                    result.append(i)
                    break
        return result

    def is_done(self):
        return self._done
