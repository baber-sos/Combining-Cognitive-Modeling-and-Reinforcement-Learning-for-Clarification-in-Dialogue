from magis.datasets.color_reference_2017.vectorized import make_or_load_cic
import heapq


class Parser():
    def __init__(self, vocabulary, topk=5):
        self._vocab = vocabulary
        self._topk = topk
    
    def parse(self, utterance, speaker='S'):
        split_utterance = utterance.split()
        #case 1


        if 'and' in split_utterance:
            
            idx = split_utterance.index('and')
            return ('andconjuction', self.parse(' '.join(split_utterance[:idx])), \
                self.parse(' '.join(split_utterance[idx + 1:])))
                
        elif ',' in utterance:

            idx = utterance.index(',')
            return ('andconjuction', self.parse(utterance[:idx]), \
                self.parse(utterance[idx + 1:]))

        elif 'not' in split_utterance:

            idx = split_utterance.index('not')
            return (self.parse(' '.join(split_utterance[:idx])), 'negation', \
                self.parse(' '.join(split_utterance[idx + 1:])) )
        
        elif 'no' in split_utterance:
            
            idx = split_utterance.index('no')
            return (self.parse(' '.join(split_utterance[:idx])), 'negation', \
                self.parse(' '.join(split_utterance[idx + 1:])) )

        elif 'or' in split_utterance:
            
            idx = split_utterance.index('or')
            return ('orconjuction', self.parse(' '.join(split_utterance[:idx])), \
                self.parse(' '.join(split_utterance[idx + 1:])))

        elif 'closest' in split_utterance:

            idx = split_utterance.index('closest')
            return ('closest', self.parse(' '.join(split_utterance[idx + 1 : ])))
            
        elif '?' in utterance:

            idx = split_utterance.index('?')
            return ('question', self.parse(utterance[:idx]))
        
        elif 'more' in split_utterance:

            idx = split_utterance.index('more')
            return ('more', self.parse(' '.join(split_utterance[idx + 1 :])))
        
        elif 'less' in split_utterance:

            idx = split_utterance.index('less')
            return ('less', self.parse(' '.join(split_utterance[idx + 1 :])))

        
        return ('cterms', self.lookup_color_term(utterance))
    
    def lookup_color_term(self, utterance):
        
        potential_candidates = list()

        for i in range(len(self._vocab)):
            color_term = self._vocab.lookup_index(i)
            color_term = color_term.split()
            
            similarity = 0
            for clr in color_term:
                if clr in utterance:
                    similarity += 1
            
            priority = -similarity/len(color_term)
            heapq.heappush(potential_candidates, (priority, i))
            potential_candidates = potential_candidates[:10]
        
        #extract the items with max score here
        to_return = list()
        
        max_score = potential_candidates[0][0]
        lookup_dict = dict()
        for i in potential_candidates:
            if i[0] <= max_score and max_score < 0:
                color_term = self._vocab.lookup_index(i[1]).split()
                if color_term[-1] not in lookup_dict:
                    to_return.append(i[1])
                    lookup_dict[color_term[-1]] = (len(to_return) - 1, len(color_term))
                else:
                    if lookup_dict[color_term[-1]][1] < len(color_term):
                        idx = lookup_dict[color_term[-1]][0]
                        to_return[idx] = i[1]
                        lookup_dict[color_term[-1]] = (idx, len(color_term))

        return [self._vocab.lookup_index(i) for i in to_return]




if __name__ == '__main__':
    f = open('test_sentences.txt')
    cic = make_or_load_cic()
    parser = Parser(cic._color_vocab)
    for utterance in f:
        parse_struct = parser.parse(utterance)
        print(parse_struct)

