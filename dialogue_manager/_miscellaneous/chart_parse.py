import copy

def chart_parse(string, grammar):

    def predictor(state, k, grammar):
        # print(state)
        cur_state = state[0].split()
        dot_ix = state[1][1]
        print('***state:', cur_state, dot_ix)
        non_terminal_for_grammar = cur_state[dot_ix]
        set_copy = state_set[k].copy()
        for next_rule in grammar[non_terminal_for_grammar]:
            # print(next_rule)
            set_copy.add((next_rule, (k, 0), non_terminal_for_grammar))
        # print('***new state set', set_copy)
        return set_copy

    def scanner(state, k, grammar):
        dot_ix = state[1][1]
        cur_state = state[0].split()
        cur_literal = cur_state[dot_ix]
        if k >= len(string):
            return
        if cur_literal == string[k]:
            new_state_to_add = (state[0], (state[1][0], state[1][1] + 1), state[2])
            state_set[k + 1].add(new_state_to_add)
    
    def completer(state, k):
        input_pos_set = state_set[state[1][0]].copy()
        state_set_k = state_set[k].copy()

        for this_state in input_pos_set:
            dot_ix = this_state[1][1]
            cur_state = this_state[0].split()
            if cur_state[dot_ix] == state[2]:
                state_set_k.add( (this_state[0], (this_state[1][0], this_state[1][1] + 1), \
                    this_state[2]) )
        return state_set_k
    
    #init the state/chart position
    state_set = [set() for i in range(len(string) + 1)]
    #state, input position, dot position
    state_set[0].add((grammar['<P>'][0], (0, 0), None))
    
    for k in range(0, len(string) + 1):
        new_set = None
        first = True
        while first or (new_set != state_set[k] and new_set != None):
            if not first:
                state_set[k] = new_set
            first = False

            for state in state_set[k]:
                if not state[1][1] == (len(state[0].split())):
                    dot_ix = state[1][1]
                    next_elem = state[0].split()[dot_ix]
                    if next_elem[0] == '<':
                        new_set = predictor(state, k, grammar)
                    else:
                        scanner(state, k, grammar)
                else:
                    new_set = completer(state, k)
                
                if new_set != state_set[k]:
                    break
    
    return state_set
    
    


if __name__ == '__main__':
    grammar = {'<P>' : ['<S>'],
        '<S>' : ['<S> + <M>', '<M>'],
        '<M>' : ['<M> * <T>', '<T>'],
        '<T>' : ['1', '2', '3', '4']}
    
    parse = chart_parse('2 + 3 * 4'.split(), grammar)
    # print([len(x) for x in chart_parse('2 * 3 + 4'.split(), grammar)])
    sets = len('2 + 3 * 4'.split())
    for i in range(sets + 1):
        print('---------------------------------SET', i + 1, '------------------------------')
        for x in parse[i]:
            print(x)