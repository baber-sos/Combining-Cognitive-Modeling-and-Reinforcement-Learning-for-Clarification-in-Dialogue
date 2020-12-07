import re

def evaluate(intent):
    output = dict()
    evaluation = None
    intent = intent.strip()

    start = 0
    temp_match = re.search('\w+', intent)
    constituents = []
    start_indices = []

    while temp_match:
        identity = temp_match.group(0)
        constituents.append(identity)
        start_indices.append(start + temp_match.span()[0])
        start += temp_match.span()[1]
        temp_match = re.search('\w+', intent[start:])
        
    name = constituents[0]
    args = []
    arr_locations = []
    try:
        start = 0
        while start < len(intent):
            cur_str = intent[start:]
            arr_locations.append((cur_str.index('['), \
                cur_str.index(']')))
            start += arr_locations[-1][1]
    except Exception as e:
        pass

    #print(arr_locations)
    arr_index = 0
    ix = 0
    constituents = constituents[1:]
    start_indices = start_indices[1:]
    while ix < len(constituents):
        arg = constituents[ix]
        temp_arg = []
        flag = False
        while ix < len(start_indices) and \
            arr_index < len(arr_locations) and \
            start_indices[ix] > arr_locations[arr_index][0] and \
            start_indices[ix] < arr_locations[arr_index][1]:
            flag = True
            temp_arg.append(constituents[ix])
            ix += 1
        #print('after here: ', ix)

        if flag:
            args.append(temp_arg)
            flag = False
            arr_index += 1
        else:
            args.append(arg)
            ix += 1

        if arr_index < len(arr_locations) and \
            start_indices[ix] > arr_locations[arr_index][1]:
            args.append([])
            arr_index += 1
        #function call
    return args

if __name__ == '__main__':
    print(evaluate('IDENTIFY_SET(S,-1,[],blueish)'))
