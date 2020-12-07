def change_variable_name(obtained_form, var_name, var):
    # var_name is the name for output variable
    form = ''
    try:
        split_form = obtained_form.split('AND')
        output_var = set()
        input_vars = set()
        func_list = []
        variable_map = dict()
        for constituent in split_form:
            constituent = constituent.strip()
            # print('Going to split', constituent)
            func_name, args = constituent.split('(')
            # print('Split arguments:', func_name, args)
            args = args[:-1].split(',')
            for arg in args:
                name = arg.split(')')[0]
                if len(name) > 1 or (ord(name) < ord('A') or ord(name) > ord('Z')):
                    continue
                if name in output_var:
                    output_var.remove(name)
                    input_vars.add(name)
                    variable_map[name] = var
                    var = chr(ord(var) + 1)
                    if var == 'T':
                        var = chr(ord(var) + 1)
                else:
                    output_var.add(name)
            func_list.append((func_name, args))
        
        for fname, args in func_list:
            for i in range(len(args)):
                name = args[i].split(')')[0]
                if name in output_var:
                    args[i] = var_name
                elif name in input_vars:
                    args[i] = variable_map[name]
            if form == '':
                form = fname + '(' + ','.join(args) + ')'
            else:
                form += ( ' AND ' + fname + '(' + ','.join(args) + ')')
    except Exception as e:
        # print('Encountered Exception:', e)
        pass
    return form, var

def get_logical_form(vocab, rule_ix_map, ix_rule_map, cur_node, var):
    child_str = ' '.join([child.var for child in cur_node.children])
    #some keywords: Composition, Expand, Negation, Rule, Single Digits
    transition_rule = rule_ix_map[child_str]
    form = ''
    if transition_rule[0].isdigit():
        form_ix = int(transition_rule[0])
        transition_rule = rule_ix_map[ix_rule_map[form_ix]]

    transition_rule = ' '.join(transition_rule).split('AND')
    children = cur_node.children
    children_pos = 0
    flag = False
    cur_pos = 0
    for i in range(len(transition_rule)):
        split_rule = transition_rule[i].split()
        if 'Expand' in split_rule[0]:
            flag = False
            expand_name = split_rule[1][:-1] if split_rule[1][-1] == ')' else split_rule[1]
            while children_pos < len(children) and children[children_pos].var != expand_name:
                children_pos += 1
                
            obtained_form, var = get_logical_form(vocab, rule_ix_map, ix_rule_map, \
                children[children_pos], var)
            children_pos += 1
            
            #change variable name here
            #first check if we need to change the variable name
            if split_rule[0][1] == '=':
                var_name = split_rule[0][0]
                obtained_form, var = change_variable_name(obtained_form, var_name, var)
            if form == '':
                form = obtained_form
            else:
                form += (' AND ' + obtained_form)

        elif split_rule[0] == 'Rule' or flag:
            ix = 0 if flag else 1
            start = 0
            try:
                if not flag:
                    cur_pos = 0
                flag = True
                # print('Children Right Now:', [x.var for x in children])
                while 1:
                    start = 0
                    end = len(split_rule[ix])
                    # print(split_rule, 'start index:', (start, end), split_rule[ix][start:end].index('<'))
                    start_ix = split_rule[ix][start:end].index('<')
                    # print('This is the start index:', split_rule[ix][start:end], start_ix)
                    end_ix = split_rule[ix][start:end].index('>')
                    # print('This is the end index:', split_rule[ix][start:end], end_ix)
                    name = split_rule[ix][start:end][start_ix : end_ix + 1]

                    # print('Encountered Variable:', name, cur_pos, [x.var for x in children])
                    while cur_pos < len(children) and children[cur_pos].var != name:
                        cur_pos += 1

                    replacement = ' '.join([child.var for child in children[cur_pos].children])
                    if name == '<CLR>':
                        if not replacement.isdigit():
                            replacement = str(vocab[replacement])
                    # print('This is replacement')
                    # print(replacement, cur_pos)
                    split_rule[ix] = split_rule[ix][:start_ix + start] + replacement + \
                        split_rule[ix][start + end_ix + 1:]
                    
                    # start += (end_ix + 1)
                    cur_pos = cur_pos + 1
                    # print(replacement, cur_pos)
            except Exception as e:
                # print('Some exception occured:',e)
                pass

            if form == '':
                # print('This is the split Rule', split_rule, ix)
                form = split_rule[ix]
            else:
                form += ' AND ' + split_rule[ix]
    
    return form, var