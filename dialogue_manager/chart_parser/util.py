import numpy as np
import spacy
from spacy_hunspell import spaCyHunSpell

nlp = spacy.load('en_core_web_sm')
hunspell = spaCyHunSpell(nlp, 'linux')
nlp.add_pipe(hunspell)

def get_correct_sentence(sent, terminals, exp_correct_map):
    for i, word in enumerate(sent):
        doc = nlp(word)
        # print('Correct Spell:', word, doc[0]._.hunspell_spell)
        if word in terminals or doc[0]._.hunspell_spell == True:
            continue
        sent[i] = get_correct_word(sent[i], exp_correct_map)
    return sent

def get_correct_word(word, exp_correct_map):
    if word not in exp_correct_map:
        return word
    possible_set = list(exp_correct_map[word])
    if word in [term[0] for term in possible_set]:
        return word
    else:
        ix = np.argmax([term[1] for term in possible_set])
        return possible_set[ix][0]

def preprocess(sentence, vocabulary):
    start = 0
    result = []
    while start < len(sentence):
        max_length = 0
        term_ix = -1
        for i in range(0, 3):
            cur_term = ' '.join(sentence[start:start + i + 1])
            try:
                term_ix = vocabulary[cur_term]
                max_length = i + 1
            except Exception as e:
                pass
        if max_length == 0:
            result.append(sentence[start])
            start += 1
        else:
            result.append(str(term_ix))
            start += max_length
    return result

def compute_first_token(first_set, grammar, nterminals, token):
    if token in first_set:
        return first_set[token]
    else:
        first_set[token] = set([])
    
    for rule, _ in grammar[token]:
        left_most = rule.split()[0].strip()

        if left_most not in nterminals:
            variations = [left_most + x for x in ['er', 'ish', 'est', 'der', 'dish', '']]
            for term in variations:
                first_set[token].add(term)
        else:
            first_set[token] |= compute_first_token(first_set, grammar, nterminals, left_most)
    return first_set[token]

def compute_first(grammar, nterminals):
    first_set = dict()
    for term in nterminals:
        if term not in first_set:
            first_set[term] = compute_first_token(first_set, grammar, nterminals, term)
    return first_set

def load_grammar(clr_vocab, cfg_path):
    #takes a file contatining the grammar rules as input. After that it converts those rules to
    #valid form which can be used by the chart parser.
    #It returns a set of terminals and non-terminals as well.
    with open(cfg_path) as cfg_file:
        grammar = dict()
        terminals = set()
        nonterminals = set()
        nonterminals.add('<P>')

        key = None
        for line in cfg_file:
            line = line.strip()

            if line.strip() == '':
                continue

            rules = list()
            if '->' in line:
                #new parsing rule discovered
                lsplit = line.split('->')
                key = lsplit[0].strip()
                grammar.setdefault(key, [])
                line = lsplit[1]

            # print(line)
            # print('Last Index:', line[-1])

            if line[-1] == '|':
                line = line[:-1]

            value = [(rule.strip(), -1) for rule in line.strip().split('|')]
            # print(value)
            
            # if key == '':
            #     print(value, '' in value)

            if '' in value:
                value.remove('')
            grammar[key] += value
            
            for rule in value:
                for token in rule[0].split():
                    token = token.strip()
                    if token[0] == '<':
                        nonterminals.add(token)
                    else:
                        for suffix in ['er', 'ish', 'est', 'der', 'dish', '']:
                            terminals.add(token + suffix)
            
    #assign probabilities
    for item in grammar:
        # print('keys:', item)
        prob = 1/len(grammar[item])
        for i in range(len(grammar[item])):
            grammar[item][i] = (grammar[item][i][0], prob)

    # print(grammar['<CLR>'])
    # clr_value = [(clr_vocab.lookup_index(i), 1/len(clr_vocab)) for i in range(len(clr_vocab))]
    clr_value = [(str(i), 1/len(clr_vocab)) for i in range(len(clr_vocab))]
    # I can extend the color grammar here or I can just the parser rule when it is matching a
    # a color phrase. I can checking by adding ish/er/ist/ at the end of color to see if it makes 
    # sense.
    grammar['<CLR>'] = clr_value

    # for i in range(len(clr_vocab)):
    #     for term in clr_vocab.lookup_index(i).split():
    #         for suffix in ['er', 'ish', 'est', 'der', 'dish', '']:
    #             terminals.add(term + suffix)

    for i in range(len(clr_vocab)):
        terminals.add(str(i))
    
    first_set = compute_first(grammar, nonterminals)
    return grammar, terminals, nonterminals, first_set

def load_grammar_intent_map(file_name):
    pos_rule = dict()
    rule_pos = dict()
    with open(file_name) as gram_intent_file:
        for line in gram_intent_file:
            line = line.strip().split()
            arrow_ix = -1
            try:
                arrow_ix = line.index('->')
            except Exception as e:
                continue
            pos_rule[int(line[0])] = ' '.join(line[1:arrow_ix])
            rule_pos[' '.join(line[1:arrow_ix])] = line[arrow_ix + 1 :]
    return pos_rule, rule_pos