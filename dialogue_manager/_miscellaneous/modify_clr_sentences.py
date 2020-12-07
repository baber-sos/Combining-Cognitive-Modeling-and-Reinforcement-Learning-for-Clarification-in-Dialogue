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