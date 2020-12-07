from manager.invoke_color_model import get_term_probs, get_patch_probs
import numpy as np
import re
import torch
from torch.distributions.categorical import Categorical as cat_dist
import os

np.seterr(all='print')

def normalize(prob_vector):
    return prob_vector/sum(prob_vector)

def eval_NONE(model, adj_model, task, speaker, vocab, state, shuffle_order):
    return (float(os.getenv('RL')), np.array([0.33, 0.33 , 0.33]), 'NONE')

def eval_IDENTIFY(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    clr_term, topk=1):
    if not clr_term.isdigit():
        clr_term = vocab[clr_term]

    clr_term = int(clr_term)
    task = task[speaker]
    patch_given_prev = np.array([state.distribution[i] for i in shuffle_order[speaker]])

    # if speaker == 'S':
    #     return (-0.1, patch_given_prev, 'INFORMATION')

    patch_given_current = get_patch_probs(model, task, clr_term, vocab)

    return (float(os.getenv('RL')), patch_given_current, 'INFORMATION')

def eval_ASK_CLARIFICATION(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    target, topk=1):
    return (float(os.getenv('RL')), target[1], 'ASK_CLARIFICATION')

def eval_CONFIRMATION(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    confirmation_phrase, target=None, topk=1):
    prev_dist = np.array([state.distribution[i] for i in shuffle_order[speaker]])
    try:
        any(target)
        temp = target[1] * prev_dist
        return (target[0], temp/sum(temp), 'CONFIRMATION')
    except Exception as e:
        # prev_dist = np.array([state.distribution[i] for i in shuffle_order[speaker]])
        return (float(os.getenv('RL')), prev_dist, 'CONFIRMATION')

def eval_REJECTION(model, adj_model, task, speaker, vocab, state, shuffle_order, 
    rejection_phrase, target=None):
    #previous turn so it was evaluated according to speaker permutation
    prob_given_current = (1 - np.array([state.distribution[i] for i in shuffle_order[speaker]]))
    prob_given_current = prob_given_current/sum(prob_given_current)
    try:
        any(target)
        prob_given_current = (prob_given_current * target[1])
        prob_given_current = prob_given_current/sum(prob_given_current)
    except Exception as e:
        pass
    return (float(os.getenv('RL')), prob_given_current, 'REJECTION')


def eval_SELECT(model, adj_model, task, speaker, vocab, state, shuffle_order, 
    selection):
    selection_ix = int(selection)
    
    prob_given_prev = np.array([state.distribution[i] for i in shuffle_order[speaker]])
    # sampled_ix = int(cat_dist(torch.tensor(prob_given_prev)).sample())
    # reward = -0.95
    reward = float(os.getenv('RF'))
    # if np.argmax(prob_given_prev) == 0:
    if all(task['S'][0].view(-1) == task['L'][selection_ix].view(-1)):
        # reward = 0.9
        reward = float(os.getenv('RS'))
    return (reward, prob_given_prev, 'SELECTION')


def eval_CONFIRM_SELECT(model, adj_model, task, speaker, vocab, state, shuffle_order, 
    phrase, selection):

    prob_given_prev = np.array([state.distribution[i] for i in shuffle_order[speaker]])
    reward = -0.3
    if np.argmax(prob_given_prev) == 0:
        reward = 0.3
    return (reward, prob_given_prev, 'SELECTION')
    
def eval_IDENTIFY_FROM_SET(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    adj_list, selected_set):
    return (float(os.getenv('RL')), selected_set[1], 'INFORMATION')  

def eval_IDENTIFY_SET(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    adj_list, quantifier, color_term):
    # task = task[speaker]
    patch_given_prev = np.array([state.distribution[i] for i in shuffle_order[speaker]])
    patch_given_current = np.array([0.33, 0.33, 0.33])
    if color_term != 'None':
        if not color_term.isdigit():
            color_term = vocab[color_term]
        patch_given_current = get_patch_probs(model, task[speaker], color_term, vocab)
    #ignore adjectives thing for now
    # for adj in adj_list:
    #     # print('I am in here. This is the adjective:', adj)
    #     if adj not in adj_model.embeddings:
    #         continue
    #     #adjective exists in the model
    #     all_distances = None
    #     if color_term == 'None':
    #         #do something here
    #         probs_matrix = np.ones((3,3))
    #         key = speaker + '_RGB'
    #         for i in range(3):
    #             ref_clr = task[key][i]
    #             direction = adj_model.get_compara_direction(adj, ref_clr)
    #             direction = direction/np.linalg.norm(direction)
    #             for j in range(3):
    #                 if i == j:
    #                     continue
    #                 scaled_clr = ref_clr + (direction * np.linalg.norm(task[key][j] - ref_clr))
    #                 probs_matrix[i][j] = np.linalg.norm(scaled_clr - task[key][j])
    #         all_distances = np.min(probs_matrix, axis=0)
    #         all_distances = all_distances/sum(all_distances)
    #     else:
    #         eng_term = vocab.lookup_index(int(color_term))
    #         try:
    #             direction = adj_model.get_compara_direction(adj, eng_term)
    #         except Exception as e:
    #             #color term not in the dictionary
    #             continue
    #         direction = direction/np.linalg.norm(direction)
    #         #determine the strength
    #         key = speaker + '_RGB'
    #         all_distances = []
    #         ref_clr = adj_model.cdict[eng_term.replace(' ', '')]
    #         for rgb_clr in task[key]:
    #             scaled_clr = ref_clr + (direction * np.linalg.norm(rgb_clr - ref_clr))
    #             all_distances.append(np.linalg.norm(rgb_clr - scaled_clr))

    #         all_distances = np.array(all_distances)/sum(all_distances)
    #     print('Distance Probability for for patches:', all_distances)
    #     patch_given_current = (patch_given_current * all_distances)
    #     patch_given_current = patch_given_current/sum(patch_given_current)

    return (float(os.getenv('RL')), patch_given_current, 'INFORMATION')

def eval_DISTINGUISH(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    selection):
    # if speaker == 'S':
    #     return (-0.1, np.array([state.distribution[i] for i in shuffle_order[speaker]]), 'INFORMATION')
    # print('The probabilities:', selection)
    probs = (1 - selection[1])
    probs = probs/sum(probs)
    # print('This is the result from DISTONGUISH:', probs)
    # return (-0.1, (1 - selection[1])/sum((1 - selection[1])), 'INFORMATION')
    return (float(os.getenv('RL')), probs, 'INFORMATION')

def eval_IDENTIFY_FROM_REF(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    first_selection, operation, second_selection):
    temp = None
    if operation == 'PLUS':
        temp = first_selection[1] + second_selection[1]
    elif operation == 'MULTIPLY':
        temp = first_selection[1] * second_selection[1]

    try:
        temp = temp/sum(temp)
    except Exception as e:
        temp = np.array([0.33, 0.33, 0.33])
    return (float(os.getenv('RL')), temp, 'INFORMATION')

def eval_COMPARE_REF(model, adj_model, task, speaker, vocab, state, shuffle_order, \
    conj_list, first_selection, second_selection):
    # print('The first one:', first_selection, 'The second selection:', second_selection)
    # print('Conjunction List:', conj_list, type(conj_list), conj_list)
    probs = None
    if len(conj_list) > 0 and 'or' in conj_list[0]:
        probs = 1 - ((1 - first_selection[1]) * (1 - second_selection[1]))
    else:
        # print(first_selection[1], second_selection[1])
        probs = first_selection[1] * second_selection[1]
    try:
        probs = probs/sum(probs)
    except Exception as e:
        probs = np.array([0.33, 0.33, 0.33])
    # temp = first_selection[1] + second_selection[1]
    return (float(os.getenv('RL')), probs, 'INFORMATION')

def get_intent_args(intent, output_dict):
    start = 0
    temp_match = re.search('\w+', intent)
    constituents = []
    start_indices = []

    while temp_match:
        identity = temp_match.group(0)
        constituents.append(identity)
        start_indices.append(start + temp_match.span()[0])
        start += temp_match.span()[1]
        temp_match = re.search('\w+', intent[start:])
        
    name = constituents[0]
    args = []
    arr_locations = []
    try:
        start = 0
        while start < len(intent):
            cur_str = intent[start:]
            arr_locations.append((cur_str.index('['), \
                cur_str.index(']')))
            start += arr_locations[-1][1]
    except Exception as e:
        pass

    arr_index = 0
    ix = 0
    constituents = [arg if arg not in output_dict else output_dict[arg] \
                    for arg in constituents[1:]]
    start_indices = start_indices[1:]
    while ix < len(constituents):
        temp_arg = []
        flag = False
        while ix < len(start_indices) and arr_index < len(arr_locations) and \
            start_indices[ix] > arr_locations[arr_index][0] and start_indices[ix] < arr_locations[arr_index][1]:
            flag = True
            temp_arg.append(constituents[ix])
            ix += 1
        if flag:
            args.append(temp_arg)
            flag = False
            arr_index += 1
        else:
            args.append(constituents[ix])
            ix += 1

        if arr_index < len(arr_locations) and \
            start_indices[ix] > arr_locations[arr_index][1]:
            args.append([])
            arr_index += 1
    return (name, args)

def evaluate(model, adj_model, task, speaker, vocab, state, composition, shuffle_order):
    # print('$$COMPOSITION$$:', composition)
    output_dict = dict()
    evaluation = None
    for i, intent in enumerate(composition.split('AND')[::-1]):
        name, args = get_intent_args(intent.strip(), output_dict)
        # print('$$Name:', name, 'Arguments:', args)
        if len(args) > 0 and args[0] != None:
            temp = eval('eval_' + name)(model, adj_model, task, speaker, \
                vocab, state._cur_state, shuffle_order, *args[1:])
            output_dict[args[0]] = temp[0:2]
            evaluation = temp
        else:
            evaluation = eval('eval_' + name)(model, adj_model, task, speaker, \
                vocab, state._cur_state, shuffle_order, *args[1:])

    if 'CLARIFICATION' in composition:
        final_probs = evaluation[1]
    else:
        probs_given_current = evaluation[1]
        cur_state = state._cur_state
        cur_speaker = cur_state.speaker
        while cur_state != None and cur_state.attachment_relation == 'ASK_CLARIFICATION':
            cur_state = cur_state.prev
            cur_speaker = cur_state.speaker

        probs_given_prev = cur_state.distribution
        if cur_state.speaker == speaker:
            # print('$%$%$%$%$%$%$%$%$%$%$%$%$%')
            # print('I WAS HERE!')
            # print('$%$%$%$%$%$%$%$%$%$%$%$%$%')
            probs_given_prev = np.array([cur_state.distribution[i] for i in shuffle_order[speaker]])
            
        # print('^^^^^^^^^^^^^^^^^^^^^^')
        # print('Probabilities Given Current Description:', probs_given_current)
        # print('Probabilities Given Previous Descriptions:', probs_given_prev)
        # print('^^^^^^^^^^^^^^^^^^^^^^')
        final_probs = probs_given_current
        if os.getenv('EVAL_MODE') == 'MISS' and state._cur_state.intent in [2,6,7]:
            try:
                name = 'MIS_ID' + {2: '1', 6 : '2', 7 : '3'}[state._cur_state.intent]
                prob_misun = float(os.getenv(name))
                # final_probs[final_probs <= 1e-20] = 0
                # print('Misunderstanding prob:', prob_misun, final_probs)
                final_probs = (prob_misun * 1/3) + ((1 - prob_misun) * final_probs)
                final_probs = final_probs/sum(final_probs)
            except FloatingPointError as e:
                pass

        
        try:
            final_probs = (probs_given_current * probs_given_prev)
            final_probs = final_probs/sum(final_probs)
        except Exception as e:
            final_probs = np.array([0.33, 0.33, 0.33])
            # print(e)
            # print(composition)
            # print(final_probs, probs_given_current, probs_given_prev)
    return (evaluation[0], final_probs, evaluation[2])