#from magis.datasets.color_reference_2017.vectorized import make_or_load_cic
import dataset
from dataset.cic import make_or_load_cic
from magis.utils.color import hsv2fft, rgb2hsv

from manager.manager import manager

from model.QAModel import QAModel
from model.QLearn import QLearn
from model.PQLearn import PQLearn

from state.state import INTENTS

from chart_parser.util import load_grammar
from chart_parser.util import load_grammar_intent_map
from config.config import *

from generation.gen_rules import generation

import pickle

import random
import os
import copy
import numpy as np
import torch
import colorsys

def hsl2tensor(hsl):
    print('Here:', hsl)
    hsv = hsl2hsv(*hsl)
    print('HSV:', hsv)
    hsv_vector = np.array(list(hsv))
    fft_vector = hsv2fft(hsv_vector)
    print('Fourier Transform:', fft_vector)
    return torch.FloatTensor(fft_vector)

def hsl2rgb(h, s, l):
    # for whatever reason, python swaps hsl -> hls; 
    return colorsys.hls_to_rgb(h, l, s)

def hsl2hsv(h, s, l): 
    # for whatever reason, python swaps hsl -> hls;  
    return colorsys.rgb_to_hsv(*colorsys.hls_to_rgb(h, l, s))

class task_manager():
    def __init__(self):
        self._cic = make_or_load_cic()
        self._cic.set_indexes('test')
        self.item_ix = []
        self.dmanagers = []
        self.num_actions = int(os.getenv('NUM_ACTIONS'))
        self.done = []
        # print('CFG Path:', os.getenv('CFG'))

        (self.grammar, self.terminals, self.nterminals, self.first_set) = load_grammar(self._cic._color_vocab, \
            os.getenv('CFG'))
        self.grammar = pickle.load(open(os.path.dirname(dataset.__file__) + '/pcfg_5.pkl', 'rb'))
        (self.irmap, self.rimap) = load_grammar_intent_map(os.getenv('INTENT_MAP'))
        self.color_pairs = []
        self.count = []
        self.indices = []

    def get_response(self, json, session_ix):
        if not self.done[session_ix][self.count[session_ix]]:
            this_manager = self.dmanagers[session_ix][self.count[session_ix]]
            print('This is the value that I get:', json)
            decision_info, next_move, done = this_manager.get_next_move(json)
            if decision_info == None:
                decision_info = ''
            logic_form = next_move
            print('$$$THIS IS THE RETURNED VALUE:', logic_form, done)
            self.done[session_ix][self.count[session_ix]] = done
            if 'SELECT' in logic_form:
                print('Selected:', logic_form.split(',')[-1].split(')')[0].strip()[1:-1])
                select_ix = int(logic_form.split(',')[-1].split(')')[0].strip()[1:-1])
                print('Clicked on this index:', select_ix)
                clr_pair = self.color_pairs[session_ix]
                res = False
                if all(clr_pair['L'][select_ix].view(-1) == clr_pair['S'][0].view(-1)):
                    res = True
                return {'name' : 'L', 'info': decision_info.replace('\n', '<br>'), 'text':''}, res
            else:
                return {'name' : 'L', \
                    'text' : generation(logic_form, self._cic._color_vocab),
                    'info': decision_info.replace('\n', '<br>')}, None
        else:
            return None, None
    
    def get_rgb_from_csv_index(self, csv_ix):
        colors = []
        row = self._cic._df.iloc[csv_ix]
        for t in ['target', 'alt1', 'alt2']:
            tcolor = row[t]
            tcolor = tuple((map(int, np.array(hsl2rgb(*tcolor)) * 255)))
            colors.append(tcolor)
        
        clr_tensors = [hsl2tensor(row[t]) for t in ['target', 'alt1', 'alt2']]
        shuffled_colors = [tup for tup in zip(clr_tensors, colors)]
        np.random.shuffle(shuffled_colors)

        clr_pair = {'S':clr_tensors, 'L': [shuffled_colors[i][0] for i in range(len(shuffled_colors))], \
            'S_RGB': colors, 'L_RGB' : [shuffled_colors[i][1] for i in range(len(shuffled_colors))]}
        
        dman = manager(clr_pair, None, QAModel(self.num_actions), \
            (self.grammar, self.terminals, self.nterminals, self.first_set), \
            (self.irmap, self.rimap), cic=self._cic)
        
        return colors, clr_pair, dman
    
    def get_rgb_from_condition(self, condition):
        sampled_ix, cur_batch_dict = self._cic.sample_condition(condition)
        
        colors = cur_batch_dict['rgb_colors']
        clr_tensors = cur_batch_dict['x_colors']

        shuffled_colors = [tup for tup in zip(clr_tensors, colors)]
        np.random.shuffle(shuffled_colors)

        clr_pair = {'S':clr_tensors, 'L': [shuffled_colors[i][0] for i in range(len(shuffled_colors))], \
            'S_RGB': colors, 'L_RGB' : [shuffled_colors[i][1] for i in range(len(shuffled_colors))]}
        
        #dman = manager(clr_pair, None, QAModel(self.num_actions), \
        #    (self.grammar, self.terminals, self.nterminals, self.first_set), \
        #    (self.irmap, self.rimap), cic=self._cic, batch_dict=cur_batch_dict)
        
        dman = manager(clr_pair, None, PQLearn(self.num_actions), \
            (self.grammar, self.terminals, self.nterminals, self.first_set), \
            (self.irmap, self.rimap), cic=self._cic, batch_dict=cur_batch_dict)
        
        print('@@@@@@@@@@@@@@@@@@THIS IS THE CIC INDEX:', sampled_ix)

        return colors, clr_pair, dman, sampled_ix
    
    def get_next(self, session_ix):
        if not self.done[session_ix][-1]:
            return [], None, None
        self.count[session_ix] += 1
        if self.count[session_ix] > 3:
            return 'None', 'None', 'None'
        condition = self.indices[session_ix][self.count[session_ix]] #indices contain condition in order
        colors, clr_pair, dman, ix = self.get_rgb_from_condition(condition)

        self.color_pairs[session_ix] = (clr_pair)
        self.dmanagers[session_ix].append(dman)
        self.done[session_ix].append(False)

        return colors, condition, ix

    def sample_color(self):
        #samples the color and also adds a new manager object

        # indices = [self._cic._df[self._cic._df['condition']=='far'].sample().index[0],
        #     self._cic._df[self._cic._df['condition']=='close'].sample().index[0],
        #     self._cic._df[self._cic._df['condition']=='split'].sample().index[0],
        #     self._cic._df[self._cic._df['condition']=='close'].sample().index[0]
        # ]
        conditions = ['far', 'close', 'split', 'close']
        # csv_ix = indices[0]
        condition = conditions[0]
        # self.indices.append(indices)
        self.indices.append(conditions)
        self.count.append(0)
        self.dmanagers.append([])
        self.done.append([])

        # colors, clr_pair, dman = self.get_rgb_from_csv_index(csv_ix)
        colors, clr_pair, dman, cic_ix = self.get_rgb_from_condition(condition)

        self.dmanagers[-1].append(dman)
        self.color_pairs.append(clr_pair)
        self.done[-1].append(False)

        return colors, condition, cic_ix

    def reset(self, session_ix):
        cic_item = self._cic[self.item_ix[session_ix]]
        clr_trans = cic_item['x_colors']
        colors = cic_item['rgb_colors']

        shuffled_colors = [tup for tup in zip(clr_trans, colors)]
        np.random.shuffle(shuffled_colors)

        clr_pair = {'S':clr_trans, 'L': [shuffled_colors[i][0] for i in range(len(shuffled_colors))], \
            'S_RGB': colors, 'L_RGB' : [shuffled_colors[i][1] for i in range(len(shuffled_colors))]}

        # self.color_pairs.append(clr_pair)

        dman = manager(clr_pair, None, QAModel(self.num_actions), \
            (self.grammar, self.terminals, self.nterminals, self.first_set), \
            (self.irmap, self.rimap), cic=self._cic)
        
        self.dmanagers[session_ix] = dman
        self.done[session_ix] = False

        return True
