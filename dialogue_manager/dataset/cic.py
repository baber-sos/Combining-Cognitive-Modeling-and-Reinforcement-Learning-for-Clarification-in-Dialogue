import ast
import json
import pandas as pd
from dataset.extended_vocab import ExtendedVocabulary
from magis.utils.color import hsv2fft, rgb2hsv

from adjective_model.adj_model import AdjModel

from manager.invoke_color_model import get_listener_predictions_cf, \
    get_listener_predictions_cs, lookup_listener_probability

from magis.utils.data import Context

from color_model.model import get_model_interface

import pickle
import os
import torch
import numpy as np
import colorsys
import random
from config import config

def hsl2rgb(h, s, l):
    # for whatever reason, python swaps hsl -> hls; 
    return colorsys.hls_to_rgb(h, l, s)

def hsl2hsv(h, s, l):
    # for whatever reason, python swaps hsl -> hls;  
    return colorsys.rgb_to_hsv(*colorsys.hls_to_rgb(h, l, s))

class CIC():
    def __init__(self):
        file_path = os.path.dirname(__file__)
        self._df = pd.read_csv(file_path + '/cic.csv',
            converters={
                'target': ast.literal_eval,
                'alt1': ast.literal_eval,
                'alt2': ast.literal_eval
            })
        self._df['utterance_events'] = self._df.full_text.apply(json.loads)
        # print('This is the file path:', file_path)
        # self._color_vocab = pickle.load(open(file_path + '/vocabulary.pkl', 'rb'))
        self._color_vocab = ExtendedVocabulary()
        # self.row_indices = pickle.load(open(file_path + '/row_indices', 'rb'))
        self.row_indices = list((self._df.index[(self._df['split'] == 'train') & \
            (self._df['lux_difficulty_rating'].isin([0,1,2,3]))]).map(lambda x : {'row_index' : int(x),
                                                                    'condition' : self._df.iloc[x]['condition']}))
        self._probability_path = os.getenv('ProbsD')
        # os.path.dirname(file_path) + '/cic_loaded_probs'
        self.spellcheck_map = pickle.load(open(file_path + '/spellcheck_map', 'rb'))
        self.adj_model = AdjModel()
        self._model = get_model_interface(self._color_vocab)
        self._parse_mistakes = pickle.load(open(file_path + '/parse_mistakes.pkl', 'rb'))
        self.semdial_ix = 0

    def hsl2tensor(self, hsl):
        # print('Here:', rgb)
        hsv = hsl2hsv(*hsl)
        # print('HSV:', hsv)
        hsv_vector = np.array(list(hsv))
        fft_vector = hsv2fft(hsv_vector)
        # print('Fourier Transform:', fft_vector)
        return torch.FloatTensor(fft_vector)
    
    def set_indexes(self, split_name):
        file_path = os.path.dirname(__file__)
        if split_name == 'test':
            self._parse_mistakes = pickle.load(open(file_path + '/test_parse_mistakes.pkl', 'rb'))
        elif split_name == 'train':
            self._parse_mistakes = pickle.load(open(file_path + '/parse_mistakes.pkl', 'rb'))
        self.row_indices = list((self._df.index[(self._df['split'] == split_name) & \
            (self._df['lux_difficulty_rating'].isin([0,1,2,3]))]).map(lambda x : {'row_index' : int(x),
                                                                    'condition' : self._df.iloc[x]['condition']}))

    def __len__(self):
        return len(self.row_indices)

    def sample_condition(self, condition):
        semdial_trial = [10290, 4749, 6153, 1317]
        if self.semdial_ix < 4:
            to_ret = (semdial_trial[self.semdial_ix], self.get_index_item(semdial_trial[self.semdial_ix]))
            self.semdial_ix += 1
            return to_ret
        print('Here is the condition:', condition)
        modified = [(ix, item) for ix, item in enumerate(self.row_indices)]
        sampled_ix, _ = random.choice(list(filter(lambda item : item[1]['condition'] == condition, modified)))
        # total_candidates = len(list(filter(lambda item : item['condition'] == \
        #     condition, self.row_index)))
        # sampled_ix = 13301
        return sampled_ix, self.get_index_item(sampled_ix)

    def __getitem__(self, index):
        batch_dict = self.row_indices[index]
        csv_ix = self.row_indices[index]['row_index']
        row = self._df.iloc[csv_ix]
        
        colors = []
        for t in ['target', 'alt1', 'alt2']:
            tcolor = row[t] #this is in hsl format
            rgb = tuple(map(int, np.array(hsl2rgb(*tcolor)) * 255))
            colors.append(rgb)

        # clr_trans = [self.rgb2tensor(clr) for clr in colors]

        batch_dict['rgb_colors'] = colors
        # batch_dict['x_colors'] = torch.cat([self.rgb2tensor(clr) for clr in colors], dim=0).view(1,3,-1)
        batch_dict['x_colors'] = [self.hsl2tensor(row[t]) for t in ['target', 'alt1', 'alt2']]
        batch_dict['y_utterance'] = -1
        
        return batch_dict
    
    def get_index_item(self, index):
        batch_dict = self.row_indices[index]
        # csv_ix = self.row_indices[index]['row_index']
        csv_ix = batch_dict['row_index']
        row = self._df.iloc[csv_ix]
        
        colors = []
        for t in ['target', 'alt1', 'alt2']:
            tcolor = row[t] #this is in hsl format
            rgb = tuple(map(int, np.array(hsl2rgb(*tcolor)) * 255))
            colors.append(rgb)

        # clr_trans = [self.rgb2tensor(clr) for clr in colors]

        batch_dict['rgb_colors'] = colors
        batch_dict['x_colors'] = [self.hsl2tensor(row[t]) for t in ['target', 'alt1', 'alt2']]
        # print('I reached here. Going to run the model now!')
        #target choices
        #check disk first. If found good otherwise go to the model
        model_probpath = self._probability_path + '/' + os.getenv('COLOR_MODEL') + '/' + \
                            str(batch_dict['row_index'])
        disk_flag = False
        if os.path.isfile(model_probpath) and (os.getenv('COLOR_MODEL') == 'COMPOSITE' or \
            os.getenv('COLOR_MODEL') == 'DCOMPOSITE'):

            probs = pickle.load(open(model_probpath, 'rb'))
            for i in range(3):
                batch_dict[i] = probs[i]
            # print('Probabilities Shape:', probs[i]['L'].shape)
            disk_flag = True

        if disk_flag == False:
            for i in range(3):
                permutation_to_use = [batch_dict['x_colors'][i]]
                for j in range(3):
                    if i != j:
                        permutation_to_use.append(batch_dict['x_colors'][j])
                if os.getenv('COLOR_MODEL') == 'CONSERVATIVE':
                    perm_context = Context.from_cic_batch({'x_colors' : torch.cat(permutation_to_use).view(1, 3, -1),\
                                                    'y_utterance' : torch.tensor(-1)})
                    self._model[0].set_new_context(perm_context)
                    speaker_probs, listener_probs = self._model[0].speaker_probas[0].to(os.getenv('DEVICE')), \
                                                        self._model[0].listener_probas[0].to(os.getenv('DEVICE'))
                    batch_dict.setdefault(i, dict())
                    batch_dict[i]['CS'] = self._model[0].conservative_speaker[0].to(os.getenv('DEVICE'))
                elif os.getenv('COLOR_MODEL') == 'RGC': 
                    perm_context = Context.from_cic_batch({'x_colors' : torch.cat(permutation_to_use).view(1, 3, -1),\
                                                    'y_utterance' : torch.tensor(-1)})
                    self._model[0].set_new_context(perm_context)
                    speaker_probs, listener_probs = self._model[0].speaker_probas[0].to(os.getenv('DEVICE')), \
                                                        self._model[0].listener_probas[0].to(os.getenv('DEVICE'))
                    batch_dict.setdefault(i, dict())
                    batch_dict[i]['RS'] = self._model[0].rgc_speaker[0].to(os.getenv('DEVICE'))
                elif os.getenv('COLOR_MODEL') == 'DCOMPOSITE':
                    # print('I am here')
                    perm_context = Context.from_cic_batch({'x_colors' : torch.cat(permutation_to_use).view(1, 3, -1),\
                                                    'y_utterance' : torch.tensor(-1)})
                    self._model[0].set_new_context(perm_context)
                    speaker_probs, listener_probs = self._model[0].composite_model.speaker_marginals[0].to(os.getenv('DEVICE')), \
                                                        self._model[0].composite_model.listener_marginals[0].to(os.getenv('DEVICE'))                           
                elif os.getenv('COLOR_MODEL') == 'COMPOSITE':
                    perm_context = Context.from_cic_batch({'x_colors' : torch.cat(permutation_to_use).view(1, 3, -1),\
                                                    'y_utterance' : torch.tensor(-1)})
                    speaker_probs, listener_probs = self._model[0](perm_context)
                elif os.getenv('COLOR_MODEL') == 'XKCD':
                    speaker_probs = self._model[0](*permutation_to_use)
                    listener_probs = get_listener_predictions_cf(self._model[0], *permutation_to_use)
                    print('Speaker Probas Shape:', speaker_probs.shape)
                    print('Listener Probas Shape:', listener_probs.shape)
                # print('Probability Shape: %s' % (speaker_probs.shape,))
                batch_dict.setdefault(i, dict())
                batch_dict[i]['S'] = speaker_probs
                batch_dict[i]['L'] = listener_probs

        batch_dict['y_utterance'] = -1

        return batch_dict

def make_or_load_cic():
    return CIC()
