from magis.vocab import Vocabulary
# import dataset
import pickle
import os

class ExtendedVocabulary():
    def __init__(self):
        self._vocab = pickle.load(open(os.path.dirname(__file__) + '/vocabulary.pkl', 'rb'))
        self._extended = dict()
        self._variations = ['er', 'ish', 'est', 'der', 'dish', '']
        self.extend()
    
    def all_variations(self, x, variations):
        if len(x) == 1:
            return [[x[0] + var] for var in variations]
        to_merge = self.all_variations(x[1:], variations)
        result = []
        for entity in to_merge:
            for var in variations:
                to_append = list(entity)
                to_append.insert(0, x[0] + var)
                result.append(to_append)
        return result

    def extend(self):
        for i in range(len(self._vocab)):
            term_tokens = self._vocab.lookup_index(i).split()
            # print('These are tokens for color terms:', term_tokens)
            variations = self.all_variations(term_tokens, self._variations)
            for var in variations:
                new_cterm = ' '.join(var)
#                 print(new_cterm)
                self._extended[new_cterm] = i
#             break

    def __getitem__(self, token):
        if token not in self._extended:
            raise KeyError('Implicit adds not supported')
        return self._extended[token]
    
    def lookup_index(self, index):
        if index >= len(self._vocab):
            raise KeyError('the index (%d) is not in the vocabulary' % index)
        return self._vocab.lookup_index(index)
    
    def __len__(self):
        return len(self._vocab)
