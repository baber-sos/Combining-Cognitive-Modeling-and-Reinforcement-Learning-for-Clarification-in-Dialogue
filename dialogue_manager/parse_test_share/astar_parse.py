import copy
from chart_parser.AStarQ import AStarQ
from chart_parser.node import node

#state representation
#(rule, (sentence_ix_where_match_started, dot_ix), parent_var, sentence_ix_being_matched,\
#parent_ix_in_the_rule_being_matched, my_index_in_the_rule_being_matched)

def chart_parse(string, grammar, terminals, nonterminals, first_variable):

    queue = None
    state_tracker = [dict() for k in range(len(string) + 1)]
    
    var_ix_dict = {k : {rule[0] : (j, i) for (i, rule) in enumerate(grammar[k])} \
        for (j, k) in enumerate(grammar)}
    ix_var_dict = {j : {i : (k, rule[0]) for (i, rule) in enumerate(grammar[k])} \
        for (j, k) in enumerate(grammar)}

    def predictor(state, k, grammar):
        rule, (i, dot_ix), parent_var, k, prix, rix = state.state
        parent_node = state.node
        
        rule = rule.split()

        cur_cost = state.cost
        hrst = 0
        for ri, (next_rule, prob) in enumerate(grammar[rule[dot_ix]]):
            next_state = (next_rule, (k, 0), rule[dot_ix], k, rix, ri)
            child_node = node(rule[dot_ix], [], parent_node, start=k)
            child_cost = (-k, cur_cost[1] * prob)

            state_tracker[k][(next_state, child_cost)] = child_node #q_entry.node

            queue.add_node(next_state, child_node, child_cost, hrst)
        

    def scanner(state, k, grammar):
        rule, (i, dot_ix), parent_var, k, prix, rix = state.state
        parent_node = state.node

        split_rule = rule.split()

        cur_cost = state.cost
        hrst = 0
        if k >= len(string):
            return
        variations = set([string[k] + x for x in ['er', 'ish', 'est', 'der', 'dish', '']])

        if split_rule[dot_ix] in variations:
            next_state = (rule, (i, dot_ix + 1), parent_var, k + 1, prix, rix)
            
            parent_node.children.append( node(split_rule[dot_ix], [], parent_node, \
                start=k, end=k+1 ) )
            parent_node.end = max(k + 1, parent_node.end)
            
            # state_tracker[k + 1][(next_state, (-k-1, cur_cost[1]))] = parent_node
            queue.add_node(next_state, parent_node, (-k-1, cur_cost[1]), hrst)

    def completer(state, k):
        rule, (i, dot_ix), parent_var, state_k, pprix, prix = state.state
        complete_node = state.node

        cost = state.cost
        hrst = 0
        
        for (entry,cur_cost) in state_tracker[i]:
            cur_rule, (cur_i, cur_dot), cur_parent, cur_k, _, rix = entry

            cur_node = state_tracker[i][(entry,cur_cost)]

            split_cur_rule = cur_rule.split()

            if cur_dot >= len(split_cur_rule):
                continue

            if parent_var == split_cur_rule[cur_dot]:
                cur_node = copy.deepcopy(state_tracker[i][(entry,cur_cost)])
                next_state = (cur_rule, (cur_i, cur_dot + 1), cur_parent, k, prix, rix)
                
                cur_node.children.append(complete_node)
                cur_node.end = complete_node.end
                
                # state_tracker[k][(next_state, (-k, cost[1]))] = cur_node
                queue.add_node(next_state, cur_node, (-k, cost[1]), hrst)

    start_state, prob = grammar[first_variable][0]
    start_state = (start_state, (0, 0), first_variable, 0, -1, 0)
    queue = AStarQ()
    start_node = node(first_variable, [], None, start=0, end=-1)
    state_tracker[0][(start_state, (0, -prob))] = start_node
    queue.add_node(start_state, start_node, (0, -prob), 0)

    while True:
        try:
            q_entry = queue.pop_node()
        except KeyError as e:
            print('cannot be parsed', e)
            return -1

        rule, (i, dot_ix), parent_var, k, _, rix = q_entry.state
        rule = rule.split()
        cur_cost = q_entry.cost

        if (q_entry.state, cur_cost) not in state_tracker[k]:
            state_tracker[k][(q_entry.state, cur_cost)] = q_entry.node

        if (i, dot_ix) == (0, 1) and parent_var == '<P>' and k == len(string):
            return q_entry.node

        if not (dot_ix == len(rule)):
            
            if rule[dot_ix] in nonterminals:
                predictor(q_entry, k, grammar)
            else:
                scanner(q_entry, k, grammar)
        else:
            completer(q_entry, k)
    

