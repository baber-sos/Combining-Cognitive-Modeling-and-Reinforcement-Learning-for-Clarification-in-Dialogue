import sys
import numpy as np
import random
import re
import os
import torch
from torch.distributions.categorical import Categorical

from manager.invoke_color_model import get_term_probs, get_patch_probs

def sample_patch(distribution, state):
    return np.argmax(distribution)

def get_max_term(model, task, vocab, clr_terms, op=max):
    # print('#########################')
    # print('These are the color terms:', clr_terms)
    # print('#########################')
    max_term = -1
    max_target_prob = 0 if op == max else 1
    for i, term in enumerate(clr_terms):
        probs = get_patch_probs(model, task, term, vocab)
        max_target_prob = op(probs[0], max_target_prob)
        if max_target_prob == probs[0]:
            max_term = term
    return max_term
    
def gen_task(task, index):
    res = [task[index]]
    for i in range(3):
        if i != index:
            res.append(task[index])
    return res

def gen_INVALID(model, task, state, turn, _):
    return 'NONE()'

def gen_NONE(model, task, state, turn, _):
    return 'NONE()'

def gen_IDENTIFY(model, task, state, turn, vocab, topk=1, target=True, var='A', \
                exclude=set()):
    # print(task)
    sim_flag = turn[1]
    turn = turn[0]
    # print('##I am here and this is the form:', state.form)
    
    clr_terms = set() | exclude
    cur_state = state
    while cur_state:
        form = cur_state.form
        match = re.search('[0-9]+', form)
        # print('Match before the loop:', match)
        while match != None:
            clr_terms.add(match.group(0))
            form = form[match.span()[1]:]
            match = re.search('[0-9]+', form)
        cur_state = cur_state.prev
    # print('Match after the loop:', match)
    prob_mistake = 0.0
    if turn == 'S' and state.intent == 2:
        prob_mistake = float(os.getenv('MIS_ID1'))
    elif turn == 'S' and state.intent == 6:
        prob_mistake = float(os.getenv('MIS_ID2'))
    elif turn == 'S' and state.intent == 7:
        prob_mistake = float(os.getenv('MIS_ID3'))
    
    sampled_val = random.random()

    if turn == 'S' and sampled_val <= prob_mistake and state.intent in [2, 6, 7]:
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^')
        # print('Mistake in generation')
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('Sampled Value: %f, Mistake Chance: %f, Previous Move: %d' % (sampled_val, prob_mistake, state.intent))
        task = gen_task(task, random.choice([1,2]))

    if turn == 'S' and (state.prev == None or target == False):
        # print('FIRST TURN')
        term_probs = get_term_probs(model, task, simulation=sim_flag)
        distribution = Categorical(term_probs)
        clr_term = str(int(distribution.sample()))
        if target:
            return 'IDENTIFY(T,' + clr_term + ')'
        else:
            return 'IDENTIFY(A,' + clr_term + ')'
    elif turn == 'S' and target == True and state.intent == 2:
        if random.random() <= 0.8:
            term_probs = get_term_probs(model, task, simulation=sim_flag)
            distribution = Categorical(term_probs)
            clr_term = str(int(distribution.sample()))
            return 'IDENTIFY(T,' + clr_term + ')'
        else:
            term_probs1 = get_term_probs(model, [task[1], task[0], task[2]], simulation=sim_flag)
            term_probs2 = get_term_probs(model, [task[2], task[0], task[1]], simulation=sim_flag)
            distribution1 = Categorical(term_probs1)
            distribution2 = Categorical(term_probs2)
            clr_term1 = str(int(distribution1.sample()))
            clr_term2 = str(int(distribution2.sample()))
            return 'DISTINGUISH(T,A) AND COMPARE_REF(A,[and],B,C) AND IDENTIFY(B,' + clr_term1 + \
                ') AND IDENTIFY(C,' + clr_term2 + ')'
    elif turn == 'S' and state.intent == 7:
        return gen_IDENTIFY1(model, task, state, turn, vocab, topk=1)
    elif turn == 'S' and state.intent == 6:
        sorted_ind = np.argsort(state.distribution)[::-1]
        if 0 in sorted_ind[:2]:
            return gen_IDENTIFY1(model, task, state, turn, vocab, topk=1)
        else:
            # print('$$$$$$$$ | TERMS IN CLARIFICATION DID NOT INDENTIFY TARGET | $$$$$$$$')
            return gen_IDENTIFY2(model, task, state, (turn, sim_flag), vocab, topk=1)

    if target == True and len(task) > 1:
        ix = sample_patch(state.distribution, state)
        new_task = [task[ix]]
        for i in range(len(task)):
            if i != ix:
                new_task.append(task[i])
        task = new_task

    start_var = var if target == False else 'T'
    term_probs=get_term_probs(model, task, simulation=False, turn=turn)
    term_probs = term_probs.topk(k=20, dim=0)
    #getting the non conflicting color patch here
    # print('term probs:', term_probs)
    clr_term = str(int(term_probs.indices[0]))
    for i in range(0,20):
        clr_term = str(int(term_probs.indices[i]))
        if clr_term not in clr_terms:
            break
    
    return 'IDENTIFY(' + start_var + ',' + clr_term + ')'

def gen_IDENTIFY1(model, task, state, turn, vocab, topk=1, prob_mistake=False):
    operation = max if prob_mistake == False else min
    if len(state.clr_terms):
        return 'IDENTIFY(T,' + str(get_max_term(model, task, vocab, state.clr_terms, op=operation)) + ')'
    return 'NONE(T)'

def gen_IDENTIFY2(model, task, state, turn, vocab, topk=1):
    sim_flag = turn[1]
    turn = turn[0]  
    term_probs = get_term_probs(model, task, simulation=sim_flag)
    distribution = Categorical(term_probs)
    clr_term = str(int(distribution.sample()))
    return 'IDENTIFY(T,' + clr_term + ')'
    

def gen_ASK_CLARIFICATION(model, task, state, turn, vocab, topk=1):
    sim_flag = turn[1]
    turn = turn[0]

    clr_ix = sample_patch(state.distribution, state)
    
    new_task = [task[clr_ix]]
    for i in range(len(task)):
        if i != clr_ix:
            new_task.append(task[i])
    task = new_task
    
    return 'ASK_CLARIFICATION(T, A) AND ' + \
        gen_IDENTIFY(model, task, state, (turn, False), vocab, topk, target=False)
    
def gen_ASK_CLARIFICATION0(model, task, state, turn, vocab, topk=1):
    return 'ASK_CLARIFICATION(T, A) AND NONE(A)'

def gen_ASK_CLARIFICATION2(model, task, state, turn, vocab, topk=1):
    ordered_ind = np.argsort(state.distribution)[::-1]
    vars = ['C', 'D']
    clarification_forms = []
    exclude = set()
    for i in range(2):
        form = gen_IDENTIFY(model, gen_task(task, ordered_ind[i]), state, (turn[0], False), \
                vocab, topk, target=False, var=vars[i], exclude=exclude)
        clarification_forms.append(form)
        match = re.search('[0-9]+', form)
        exclude.add(match.group(0))
    return 'ASK_CLARIFICATION(T, A) AND COMPARE_REF(A,[or],C,D) AND ' + \
                clarification_forms[0] + ' AND ' + clarification_forms[1]

def gen_ASK_CLARIFICATION3(model, task, state, turn, vocab, topk=1):
    ordered_ind = np.argsort(state.distribution)
    vars = ['C', 'D', 'E']
    clarification_forms = []
    exclude = set()
    for i in range(3):
        form = gen_IDENTIFY(model, gen_task(task, ordered_ind[i]), state, (turn[0], False), \
                vocab, topk, target=False, var=vars[i], exclude=exclude)
        clarification_forms.append(form)
        match = re.search('[0-9]+', form)
        exclude.add(match.group(0))
    form1 = ' COMPARE_REF(A,[or],C,D) AND ' + clarification_forms[0] + ' AND ' + clarification_forms[1]
    form2 = 'COMPARE_REF(B,[or],A,E) AND ' + clarification_forms[2]
    return 'ASK_CLARIFICATION(T,B) AND ' + form2 + ' AND' + form1

    
def gen_CONFIRMATION(model, task, state, turn, vocab, topk=1):
    evaluation = state.evaluation
    sim_flag = turn[1]
    turn = turn[0]
    
    if turn == 'S':
        return "CONFIRMATION(None, 'YES')"

    elif turn == 'L':
        return "CONFIRMATION(None, 'YES')"
        # return gen_CONFIRM_SELECT(model, task, state, (turn, sim_flag), vocab, topk=1)

def gen_REJECTION(model, task, state, turn, vocab, topk=1):
    if random.random() <= 0.62:
        return 'REJECTION(T,NO)'
    else:
        return 'REJECTION(T,NO, A) AND ' + \
            gen_IDENTIFY(model, task, state, turn, vocab, target=False)    
     #return 'REJECTION(None,NO, A) AND ' + \
     #    gen_IDENTIFY(model, task, state, turn, vocab, target=False)

def gen_SELECT(model, task, state, turn, vocab, topk=1):
    sim_flag = turn[1]
    turn = turn[0]
    select_mistake = False

    length = 0
    cur_state = state
    while cur_state.speaker != None:
        length += 1
        cur_state = cur_state.prev
        if length > 1:
            break

    if length == 1 and random.random() <= float(os.getenv('SELECT_MISTAKE')):
        select_mistake = True

    ix = state.evaluation
    # print(state.distribution)
    if os.getenv('MATCHER_STRAT') == 'LIST_SAMPLE':
        ix = int(Categorical(torch.tensor(state.distribution)).sample())
    elif os.getenv('MATCHER_STRAT') == 'LIST_BEST':
        ix = state.evaluation
    # return "SELECT(None, '" + str(state.evaluation) + "')"
    if select_mistake:
        ix = random.choice(list(set([0, 1, 2]) - {ix}))
    return "SELECT(None, '" + str(ix) + "')"

def gen_CONFIRM_SELECT(model, task, state, turn, vocab, topk=1):
    sim_flag = turn[1]
    turn = turn[0]

    return "CONFIRM_SELECT(None, 'OKAY', '"  + str(state.evaluation) + "')"

