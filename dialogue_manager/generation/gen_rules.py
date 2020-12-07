import random

#ASK_CLARIFICATION_0(None)
#ASK_CLARIFICATION_1(T, A) AND IDENTIFY(A,<CLR>)
#ASK_CLARIFICATION_2(T,A,B) AND IDENTIFY(A,<CLR>) AND IDENTIFY(B,<CLR>)
#IDENTIFY(A,<CLR>)
#CONFIRMATION(None, 'YES')
#CONFIRMATION(None, 'OKAY')
#REJECTION(None, 'NO', T) AND IDENTIFY(T,<CLR>) => No, its the red one

generation_map = {
    'IDENTIFY' : ["CRL", "It's the CRL"],
    'CONFIRMATION' : ["Got it", "OK!"],
    'ASK_CLARIFICATION_0' : ["I don't know.", "I'm not sure.", "humm?"],
    'ASK_CLARIFICATION_1' : ["Do you mean the <CLR_0> ?", "The <CLR_0>?",\
        "Like the more <CLR_0>?"],
    'ASK_CLARIFICATION' : {1: ["Do you mean <CLR_0> ?", "The <CLR_0>?"],
                            2: ["<CLR_0> or <CLR_1>?", "Do you mean <CLR_0> or <CLR_1>?"],
                            3: ["<CLR_0> or <CLR_1> or <CLR_2>?"]
                            },
    'ASK_CLARIFICATION_2' : ["The <CLR_0> or <CLR_1>?", "More <CLR_0> or <CLR_1>?"]
}

def get_count(lform):
    count = 0
    for constt in lform.split('AND'):
        color_term = constt.split(',')[1].split(')')[0].strip()
        if color_term.isdigit():
            count += 1
    return count

def tokenize(string):
    symbol_list = ['.', ',', '?', '/', '\\', '?', '-', ';', '!']
    utterance = string.lower()
    for s in symbol_list:
        utterance = (' ' + s + ' ').join(utterance.split(s)).strip()
    return utterance.split()

def check_greetings(logic_form):
    greetings = ['hello', 'hi', 'nice to meet you']
    for greeting in greetings:
        if greeting in logic_form:
            return True
    return False

def check_directions(logic_form):
    direction = ['middle', 'right', 'left', 'first', 'second', 'third']
    for dctn in direction:
        if dctn in logic_form:
            return True
    return False

def check_social(logic_form):
    social_talk = ['how are you', 'you are well']
    for soct in social_talk:
        if soct in logic_form:
            return True
    return False

def check_apology(logic_form):
    apologies = ['sorry', 'apologize']
    for apology in apologies:
        if apology in logic_form:
            return True
    return False

def check_social_stuff(original_lf):
    logic_form = tokenize(original_lf)
    if 'haha' in logic_form or 'haha' in original_lf:
        return 'I am glad you like it, but please provide me with a color patch description.'
    elif 'intelligent' in logic_form or 'intelligent' in original_lf:
        return 'Thank you. I beleive you are better, though. Now please provide me with a color description.'
    elif check_greetings(logic_form):
        return 'Hello. Nice to talk to you. Plaese provide me with a valid color description'
    elif check_directions(logic_form):
        return 'I do not see the same ordering of color patches as you. Please provide me with a color description.'
    elif check_social(logic_form):
        return 'Hi. Please provide me with a color description.'
    elif check_apology(logic_form):
        return 'No worries. Please lets continue the task.'
    else:
        return ''

def generation(logic_form, vocab):
    print('^^^^^^^^^^^^^^^^^^^^^^')
    print('This is the logic form:', logic_form)
    print('^^^^^^^^^^^^^^^^^^^^^^')
    if 'MAX' in logic_form:
        return 'This task has exceeded maximum conversation length. Please exit ' + \
            'if you have attemped all 4 tasks, or continue to the next task. Thank you.'
    elif 'NONE' in logic_form:
        return 'I could not understand you.'
    elif 'ASK_CLARIFICATION_0' in logic_form:
        sampling = random.choices(generation_map['ASK_CLARIFICATION_0'])[0]
        return sampling
    elif 'ASK_CLARIFICATION_1' in logic_form:
        sampling = random.choices(generation_map['ASK_CLARIFICATION_1'])[0]
        color_term = logic_form.split(',')[-1].split(')')[0].strip()
        return sampling.replace('<CLR_0>', color_term)
    elif 'ASK_CLARIFICATION' in logic_form:
        count = 0
        num_cp = get_count(logic_form)
        sampling = random.choices(generation_map['ASK_CLARIFICATION'][num_cp])[0]
        for constt in logic_form.split('AND'):
            color_term = constt.split(',')[1].split(')')[0].strip()
            if color_term.isdigit():
                color_term = vocab.lookup_index(int(color_term))
                sampling = sampling.replace('<CLR_' + str(count) + '>', color_term)
                print('This is the color term: %s' % (color_term,))
                print('Sampled Sentence: %s' % (sampling,))
                count += 1
        return sampling
    elif 'haha' in logic_form:
        return 'I am glad you like it, but please provide me with a color patch description.'
    elif check_greetings(logic_form):
        return 'Hello. Nice to talk to you. Plaese provide me with a valid color description'
    elif check_directions(logic_form):
        return 'I do not see the same ordering of color patches as you. Please provide me with a color description.'
    elif check_social(logic_form):
        return 'Hi. Please provide me with a color description.'
    elif check_apology(logic_form):
        return 'No worries. Please lets continue the task.'
    else:
        return 'I could not understand you.'



# print(generation('ASK_CLARIFICATION_1(T, A) AND IDENTIFY(A,PURPLE)'))
