import sys
sys.path.insert(0, '/home/sos/CIC/system/dialogue_manager')
# sys.path.insert(0, '/ilab/users/bk456/Dialogue_Research/color_in_context/system/dialogue_manager')
import pickle
import numpy as np
import copy
import random

import torch

from manager.manager import manager

from model.QAModel import QAModel
from model.QLearn import QLearn
from model.PQLearn import PQLearn
from model.SQAModel import SQAModel

from chart_parser.util import load_grammar
from chart_parser.util import load_grammar_intent_map

import dataset
from dataset.cic import make_or_load_cic

from state.state import INTENTS

import config.config

import os

np.seterr(all='raise', under='print')
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

def check_policy(action_choices):
    encountered = []
    count = 0
    for item in action_choices:
        if item[1] in [2,6,7] and item[1] not in encountered:
            encountered.append(item[1])
            count += 1
    return count

if __name__ == '__main__':
    #speaker, listener
    rl_flag = [False, True]

    cic = make_or_load_cic()
    (g, t, n, fs) = load_grammar(cic._color_vocab, os.getenv('CFG'))
    g = pickle.load(open(os.path.dirname(dataset.__file__)  + '/pcfg_5.pkl', 'rb'))
    (irmap, rimap) = load_grammar_intent_map(os.getenv('INTENT_MAP'))

    # item = cic[0]
    
    models = []
    num_actions = len(INTENTS)
    print('THIS IS NUMBER OF ACTIONS:', num_actions)
    # for i in range(2):
    #     if not rl_flag[i]:
    #         models.append(QAModel(num_actions))
    #     else:
    #         models.append(PQLearn(int(num_actions)))
    #lr = [0.01, 0.001, 0.0001, 0.00001]
    #num_epochs = [2,3,4]
    #weight_decay = [10**(-6), 10**(-4), 10**(-2)]
    num_successes = -1
    best_model_reward = -1 * float('inf')
    #open log file
    with open('reward_log', 'w') as lf:
        lf.write('Training Start\n')
    for m in range(int(os.getenv('NUM_MODELS'))):
        new_model = True
        #models = [QAModel(num_actions), PQLearn(int(num_actions))]
        models = [QAModel(num_actions), PQLearn(int(os.getenv('NUM_ACTIONS')))]
        prev_test_error = 0
        max_avg_reward = -1 * float('inf')
        loss_counter = 0
        for turns in range(int(os.getenv('NUM_EPOCHS'))):
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            # os.environ['DEVICE'] = 'cuda:0'

            print('$$$$$$$$$$$$$$$$$$$$$$$$')
            print('epoch', turns, ' started')
            start = int(os.getenv('START_INDEX'))
            end = start + int(os.getenv('NUM_CONVERSATIONS'))
            for i in range(start, end):
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
                if i < len(cic):# and i not in cic._parse_mistakes:
                    # ix = random.randint(0, len(cic) - 1)
                    #i = random.randint(0, len(cic) - 1)
                    item = cic.get_index_item(i)

                    if (os.getenv('SIMULATOR') == 'SAMPLE') or (i in cic._parse_mistakes):
                        models[0] = SQAModel(num_actions, cic=cic, ix=i)
                    elif os.getenv('SIMULATOR') == 'MODEL':
                        models[0] = QAModel(num_actions)

                    colors = item['rgb_colors']
                    clr_trans = item['x_colors']

                    shuffled_colors = [tup for tup in zip(clr_trans, colors)]
                    np.random.shuffle(shuffled_colors)

                    clr_pair = {'S':clr_trans, 'L': [shuffled_colors[i][0] for i in range(len(shuffled_colors))], \
                        'S_RGB': colors, 'L_RGB' : [shuffled_colors[i][1] for i in range(len(shuffled_colors))]}

                    dman = manager(clr_pair, *models, (g, t, n, fs), \
                        (irmap, rimap), cic=cic, batch_dict=item)
                    speaker = 'S'
                    u , lf, done = dman.get_next_move()
                    # print(speaker + ':', lf)
                    while not done:
                        # speaker = 'L' if speaker == 'S' else 'S'
                        u, lf, done = dman.get_next_move()
                        # print(speaker + ':', lf)
                    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

            print('############################')
            if rl_flag[1] == True:
                print('Steps done by RL Model:', models[1].steps_done)
            print('Total Steps:', os.getenv('EPS_DECAY'))
            print('STARTING TEST PHASE')
            print('############################')

            for model in models:
                model.set_test_phase(True)
            # print(models[1].train_loss_track)
            # exit()
            print('############################')
            num_convos = int(os.getenv('NUM_CONVERSATIONS'))
            test_convos = int(os.getenv('TEST_CONVERSATIONS')) + num_convos
            cic.set_indexes('test')
            print('Starting test phase:', num_convos, test_convos)
            # exit()
            for i in range(0, int(os.getenv('TEST_CONVERSATIONS'))):
                if i < len(cic):
                    item = cic.get_index_item(i)

                    if (os.getenv('SIMULATOR') == 'SAMPLE') or (i in cic._parse_mistakes):
                        models[0] = SQAModel(num_actions, cic=cic, ix=i)
                    elif os.getenv('SIMULATOR') == 'MODEL':
                        models[0] = QAModel(num_actions)

                    colors = item['rgb_colors']
                    # print(colors)
                    clr_trans = item['x_colors']

                    shuffled_colors = [tup for tup in zip(clr_trans, colors)]
                    np.random.shuffle(shuffled_colors)
                    clr_pair = {'S':clr_trans, 'L': [shuffled_colors[i][0] for i in range(len(shuffled_colors))], \
                        'S_RGB': colors, 'L_RGB' : [shuffled_colors[i][1] for i in range(len(shuffled_colors))]}
                    
                    dman = manager(clr_pair, *models, (g, t, n, fs), \
                        (irmap, rimap), cic=cic, batch_dict=item)
                    speaker = 'S'
                    u, lf, done = dman.get_next_move()
                    # print(speaker + ':', lf)
                    while not done:
                        speaker = 'L' if speaker == 'S' else 'S'
                        u, lf, done = dman.get_next_move()
                        # print(speaker + ':', lf)
                # print('\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%')
            
            #uncomment later
            cur_test_loss = sum(models[1].loss_track)/len(models[1].loss_track)
            if cur_test_loss < prev_test_error:
                change = False
            elif abs(cur_test_loss - prev_test_error) < 0.01:
                change = False
            else:
                change = True

            prev_test_error = cur_test_loss

            for model in models:
                model.set_test_phase(False, change=change)
                # model.set_test_phase(False, change=True)
            
            cur_avg_reward = sum(models[1].test_reward_track[-1])/len(models[1].test_reward_track[-1])
            cic.set_indexes('train')

            policy_count = check_policy(models[1].test_action_choices[-1])

            log_file = open('reward_log', 'a')
            log_file.write('Model Number: %d, Avg Reward: %.3f, Avg Loss: %.3f, # Clarifications Used: %d\n' % \
                            (m + 1, cur_avg_reward, cur_test_loss, policy_count))
            log_file.close()

            # if new_model == True and \
            #     (cur_avg_reward < best_model_reward or \
            #     cur_avg_reward < float(os.getenv('THRESH_REWARD'))):
            #     break
            # else:
            #     new_model = False
            # if policy_count < 3:
            #     if turns >= 2:
            #         break
            #     continue
            # if new_model == True:
            #     if cur_avg_reward > float(os.getenv('THRESH_REWARD')):
            #         new_model = False
            #         pass
            #     else:
            #         break
            #if new_model == True and policy_count < 2:
            #    break

            if cur_avg_reward >= max_avg_reward:
                max_avg_reward = cur_avg_reward
                if max_avg_reward >= best_model_reward:
                    best_model_reward = max_avg_reward
                    models[1].save_model()
                loss_counter = 0
            else:
                loss_counter += 1
            
            #uncomment later
            # if new_model == True:
            #     if cur_avg_reward >= max_avg_reward and \
            #         cur_avg_reward >= float(os.getenv('THRESH_REWARD')):
            #         max_avg_reward = cur_avg_reward
            #     else:
            #         break
            #     new_model = False
            # else:
            #     max_avg_reward = max(max_avg_reward, cur_avg_reward)
                
            # for i, model in enumerate(models):
            #     if i == 1:
            #         model.save_model()
            if loss_counter >= int(os.getenv('BREAK_CONDITION')):
                break
        
    # for i, model in enumerate(models):
    #     if i == 1:
    #         model.plot_statistics()
