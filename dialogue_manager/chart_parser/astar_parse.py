from chart_parser.AStarQ import AStarQ
from chart_parser.node import node

def make_copy(parse_tree):
    node_copy = node(parse_tree.var, [], parse_tree.parent, start=parse_tree.start, end=parse_tree.end)
    for child in parse_tree.children:
        node_copy.children.append(make_copy(child))
    return node_copy

def check_if_in_first(variable, nonterminals, terminals, first_set, k, string):
    # return False
    # global ignored_count
    if k < len(string) and (variable in nonterminals) and \
        (string[k] not in first_set[variable]):
        return True
    return False

def initialize_state(start_k, first_variable, state_tracker, grammar):
    start_state, prob = grammar[first_variable][0]
    
    start_node = node(first_variable, [], None, start=start_k, end=-1)
    
    start_state = (start_state, (start_k, 0), first_variable, start_k, ('<P>',))
    start_cost = (-start_k, -prob) #-prob
    
    state_tracker[start_k][(start_state, start_cost)] = start_node
    
    queue = AStarQ()
    queue.add_node(start_state, start_node, start_cost, 0)
    
    max_node = (start_node, (-start_k, -prob))
    
    return queue, state_tracker, max_node

#state representation
#(rule, (sentence_ix_where_match_started, dot_ix), parent_var, sentence_ix_being_matched,\
#parent_ix_in_the_rule_being_matched, my_index_in_the_rule_being_matched)
# global ignored_count
# global total_count
# ignored_count = 0
# total_count = 0

def chart_parse(string, grammar, terminals, nonterminals, first_set, first_variable):
    # print('Received String:', string)
    queue = None
    state_tracker = [dict() for k in range(len(string) + 1)]
    max_node = tuple()
    partial_trees = list()

    def predictor(state, k, grammar):
        if k < len(string) and string[k] not in terminals:
            return
        elif k >= len(string):
            return
        rule, (i, dot_ix), parent_var, k, cur_tree = state.state
        parent_node = state.node
        
        rule = rule.split()

        if check_if_in_first(rule[dot_ix], nonterminals, terminals, first_set, k, string):
            return

        cur_cost = state.cost
        hrst = 0
        for ri, (next_rule, prob) in enumerate(grammar[rule[dot_ix]]):
            left_most = next_rule.split()[0].strip()
            
            if check_if_in_first(left_most, nonterminals, terminals, first_set, k, string):
                continue
            
            next_state = (next_rule, (k, 0), rule[dot_ix], k, (rule[dot_ix],))
            child_node = node(rule[dot_ix], [], parent_node, start=k)
            child_cost = (-k, -prob)

            state_tracker[k][(next_state, child_cost)] = child_node 
            queue.add_node(next_state, child_node, child_cost, hrst)
        

    def scanner(state, k, max_node):
        rule, (i, dot_ix), parent_var, k, cur_tree = state.state
        parent_node = state.node

        split_rule = rule.split()
            
        cur_cost = state.cost
        hrst = 0
        if k >= len(string):
            return max_node
        variations = set([split_rule[dot_ix] + x for x in ['er', 'ish', 'est', 'der', 'dish', '']])

        if string[k] in variations:
            next_state = (rule, (i, dot_ix + 1), parent_var, k + 1, (*cur_tree, (split_rule[dot_ix],)) )
            parent_node = make_copy(parent_node)
            parent_node.children.append( node(split_rule[dot_ix], [], parent_node, \
                start=k, end=k+1) )
            parent_node.end = max(k + 1, parent_node.end)
            
            if parent_node and parent_node.end > max_node[0].end:
                max_node = (parent_node, (-k-1, cur_cost[1]))
            elif parent_node and parent_node.end == max_node[0].end and (-k-1, cur_cost[1]) < max_node[1]:
                max_node = (parent_node, (-k-1, cur_cost[1]))
            #one look ahead
            if dot_ix + 1 < len(split_rule):
                if check_if_in_first(split_rule[dot_ix + 1], nonterminals, terminals,\
                    first_set, k + 1, string):
                    return max_node
            state_tracker[k + 1][(next_state, (-k-1, cur_cost[1]))] = parent_node
            queue.add_node(next_state, parent_node, (-k-1, cur_cost[1]), hrst)
        return max_node

    def completer(state, k, max_node):
        rule, (i, dot_ix), parent_var, state_k, complete_tree = state.state
        complete_node = state.node

        cost = state.cost
        hrst = 0
        
        for (entry,cur_cost) in list(state_tracker[i].keys()):
            cur_rule, (cur_i, cur_dot), cur_parent, cur_k, new_parenttree = entry
            cur_node = state_tracker[i][(entry,cur_cost)]
            split_cur_rule = cur_rule.split()
            if cur_dot >= len(split_cur_rule):
                continue
            
            if parent_var == split_cur_rule[cur_dot]:
                if (cur_dot+1) < len(split_cur_rule):
                    if check_if_in_first(split_cur_rule[cur_dot + 1], nonterminals, terminals,\
                        first_set, k, string):
                        continue
                
                cur_node = make_copy(state_tracker[i][(entry,cur_cost)])
                next_state = (cur_rule, (cur_i, cur_dot + 1), cur_parent, k, (*new_parenttree, complete_tree))

                cur_node.children.append(complete_node)
                cur_node.end = complete_node.end
                complete_node.parent = cur_node
                if cur_parent == '<P>' and cur_node.end == max_node[0].end and \
                    (-k, cur_cost[1]) <= max_node[1]:
                    max_node = (cur_node, (-k, cost[1]))
                # print(cost, cur_cost)
                new_cost = -1 * abs(cost[1] * cur_cost[1])
                state_tracker[k][(next_state, (-k, new_cost))] = cur_node
                queue.add_node(next_state, cur_node, (-k, new_cost), hrst)
                
        return max_node

    start_k = 0
    
    queue, state_tracker, max_node = initialize_state(start_k, first_variable, state_tracker, grammar)
    flag = False
    
    #####
    # global total_count
    while True:
        try:
            q_entry = queue.pop_node()
        except KeyError as e:
            # print('cannot be parsed')
            # print('best tree till now:', max_node[0].to_str(), (max_node[0].start, max_node[0].end))
            start_k = -1
            
            if max_node[0].end <= 0:
                partial_trees.append(None)
                start_k = max_node[0].start + 1
            elif max_node[0].var != '<P>':
                partial_trees.append(None)
                start_k = max_node[0].end
            else:
                partial_trees.append(max_node[0])
                start_k = max_node[0].end

            while (start_k < len(string)) and (string[start_k] not in first_set[first_variable]):
                start_k += 1
            
            if start_k >= len(string):
                return partial_trees
            
            queue, state_tracker, max_node = initialize_state(start_k, first_variable, state_tracker, grammar)
            q_entry = queue.pop_node()
            
        rule, (i, dot_ix), parent_var, k, cur_tree = q_entry.state

        rule = rule.split()
        cur_cost = q_entry.cost

        if (q_entry.state, cur_cost) not in state_tracker[k]:
            state_tracker[k][(q_entry.state, cur_cost)] = q_entry.node
    
        if q_entry.node.var == '<P>' and q_entry.node.start == 0 and q_entry.node.end == len(string):
            q_entry.node.print_tree()
            return [q_entry.node]
        
        if not (dot_ix == len(rule)):
            if rule[dot_ix] in nonterminals:
                predictor(q_entry, k, grammar)
            else:
                max_node = scanner(q_entry, k, max_node)
        else:
            max_node = completer(q_entry, k, max_node)