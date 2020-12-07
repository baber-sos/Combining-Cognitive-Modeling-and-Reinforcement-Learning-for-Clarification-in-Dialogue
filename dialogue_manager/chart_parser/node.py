import math

class node():
    def __init__(self, var=None, children=[], parent=None, start=-1, end=-1):
        self.var = var
        self.children = children
        self.parent = parent
        self.start = start
        self.end = end

    def is_leaf(self):
        if self.var[0] == '<':
            return False
        return len(self.children) == 0
    
    def __str__(self):
        if self.parent:
            return ('Node: %s \t| Children: %s \t| Parent: %s' % (self.var, \
                ' '.join([x.var for x in self.children]), self.parent.var))
        else:
            return ('Node: %s \t| Children: %s \t| Parent: None' % (self.var, \
                ' '.join([x.var for x in self.children])))
    
    def check_useless(self):
        if self.is_leaf():
            return False
        flag = False
        for gc in self.children:
            check = gc.check_useless()
            # print(check)
            flag = flag or check
        
        return flag

    def remove_useless(self, interval):
        # print(self)
        def find_ranges(ranges, intv):
            to_return = []
            for (i, rng) in enumerate(ranges):
                if rng[1] == (intv[0] - 1):
                    to_return.append(i)
            return to_return

        ranges = []
        for i, child in enumerate(self.children):
            this_interval = (child.start, child.end)
            cur_ranges = find_ranges(ranges, this_interval)
            
            if cur_ranges == []:
                ranges.append(([i], this_interval))

            for to_modify in cur_ranges:
                ranges[to_modify][0].append(i)
                ranges[to_modify] = (ranges[to_modify][0], \
                    (ranges[to_modify][1][0], this_interval[1]))
        # print(ranges)
        for child_list, intv in ranges:
            # if self.children[childi] == 
            if intv != interval:
                for childi in child_list:
                    self.children[childi] = None
            else:
                # print(child_list)
                for childi in child_list:
                    self.children[childi].remove_useless((self.children[childi].start, \
                        self.children[childi].end))
        # print(self.children)

    def to_str(self):
        children = [self.var]
        if '<' in self.var:
            children = [self.var[1:-1]]
        for child in self.children:
            children.append(child.to_str())
        return tuple(children)
    
    def print_tree(self):
        # self.remove_useless((self.start, self.end))
        # return

        # for ch in self.children:
        #     print(ch.var, ch.start, ch.end)
        
        queue = [([self], 0)]
        to_print = []
        lvl_2_wdth = {0: 1}
        max_width = 0

        while len(queue) != 0:
            cur_popped, level = queue.pop()
            if level >= len(to_print):
                to_print.append([])
            to_print[level].append([i.var+'['+str(i.start)+','+str(i.end)+']'  for i in cur_popped])
            for child in cur_popped:
                childtoq = []
                for gc in child.children:
                    childtoq.append(gc)
                queue.insert(0, (childtoq, level + 1))
                if level + 1 not in lvl_2_wdth:
                    lvl_2_wdth[level + 1] = 0
                len_to_add = len(childtoq) if len(childtoq) > 0 else 1
                lvl_2_wdth[level + 1] += len_to_add
                max_width = max(max_width, lvl_2_wdth[level + 1])
        
        for level in range(len(to_print)):
            lvl_str = ''
            cnst_factor = 4
            cur_width = lvl_2_wdth[level]
            spaces_to_add = math.ceil((max_width * cnst_factor + 1)/(cur_width + 1))
            lvl_str += (' ' * spaces_to_add)
            for cg in to_print[level]:
                # lvl_str += '['
                for i, c in enumerate(cg):
                    if i != 0:
                        lvl_str += ((' ' * spaces_to_add) + c)
                    else:
                        lvl_str += ('[' + c)
                if len(cg) == 0:
                    lvl_str += '['
                lvl_str += ']' + (' ' * spaces_to_add)
        
            # print(lvl_str)
    
    # def traverse(self, string, level=0, ix=(0, 0)):
    #     done = False
    #     if (level) >= len(string):
    #         string.append([])
    #         string.append([])
        
    #     if ix[1] >= len(string[level]):
    #         string[level].append([])
    #         string[level + 1].append([])
        
    #     string[level][ix[1]].append((self.var, ix[0]))
    #     for i in range(len(self.children)):
    #         string[level + 1][ix[1]].append('/\t')

    #     if len(self.children) == 0:
    #         return string
        
    #     for i, child in enumerate(self.children):
    #         child.traverse(string, level + 2, (ix[1], i))

    #     return string

            # print(level)
        # print(to_print)
        
        # tree = self.traverse([])
        # level_count = []
        # width = 0
        # for i in range(len(tree)):
        #     count = 0
        #     for j in range(len(tree[i])):
        #         print(tree[i][j])
        #         count += len(tree[i][j])
        #     level_count.append(count)
        # width = max(level_count)
        
        # to_print = ''
        # for i in range(len(tree)):
        #     temp_str = []
        #     for j in range(len(tree[i])):
        #         if i % 2 == 1:
        #             continue
        #         for k in range(len(tree[i][j])):
        #             node_name, ix = tree[i][j][k]
        #             while ix >= len(temp_str):
        #                 temp_str.append('')
        #             temp_str[ix] += node_name + '\t'
        #             for x in range(len(temp_str)):
        #                 if temp_str[x] == '':
        #                     temp_str[x] = 'NC'
        #     print(' | '.join(temp_str))
                

        # print(width)

        # if len(self.chi)
        # under_consideration = [self.children]
        # print(self.var)
        # while not done:
        #     temp = []
        #     count = 0
        #     for group in under_consideration:
        #         for child in group:
        #             print(child.var, end='|')
        #             temp.append(child.children)
        #             count += len(child.children)
        #         print('&&|', end='')
        #     print()

        #     if count == 0:
        #         break
        #     else:
        #         under_consideration = temp
            
            # break
    # def __init__(self, var, parent, children):
    #     self.var = va