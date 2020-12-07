import sys
sys.path.insert(0, '/home/sos/Dialogue_Research/color_in_context/system/dialogue_manager')

# from magis.datasets.color_reference_2017.vectorized import make_or_load_cic
from dataset.cic import make_or_load_cic

from chart_parser.astar_parse import chart_parse
from chart_parser.util import load_grammar
from chart_parser.util import load_grammar_intent_map
from chart_parser.util import preprocess
from chart_parser.logic import get_logical_form
import chart_parser
import os

import cProfile as prof

if __name__ == '__main__':
    cic = make_or_load_cic()
    dir_name = os.path.dirname(chart_parser.__file__)
    g, t, n, fs = load_grammar(cic._color_vocab, dir_name + '/syntaxparsingrules_V3')
    print(g['<adj>'])
    irmap, rimap = load_grammar_intent_map(dir_name + '/gram_intent_map')
    print(irmap)
    print(rimap)

    print('and' in fs['<S>'],'and' in fs['<ConjP>'], 'and' in fs['<conj>'])
    print('Length of Terminals:', len(t), 'Terminals:', t)

    with open('test_sentences.txt') as f:
        for line in f:
            print('To Parse:', line.strip())
            final_form = ''
            parse_tree = chart_parse(preprocess(line.strip().split(), cic._color_vocab), g, t, n, fs, '<P>')
            for p in parse_tree:
                if p:
                    p.print_tree()
                print('Logical Form:', get_logical_form(cic._color_vocab, rimap, irmap, p, 'E'))
            print('-----------------')
