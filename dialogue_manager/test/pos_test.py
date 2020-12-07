import spacy
from magis.datasets.color_reference_2017.vectorized import make_or_load_cic

set_classifications = set()
cic = make_or_load_cic()
df = cic._df

colors = ['blue', 'red', 'black', 'blueish', 'purpleish']

# nlp = spacy.load('en_core_web_lg')

# for ci in range(len(cic._color_vocab)):
#     color = cic._color_vocab.lookup_index(ci)
#     # print(color)
#     doc = nlp(color)
#     res = ''
#     for token in doc:
#         res += token.tag_
#         if token.tag_ == 'JJ':
#             print(color, ',', token)
#         res += ' '
#     set_classifications.add(res[:-1])
    # print('-----------------------')


for i in range(1, 5000):
    if len(df['utterance_events'].iloc[i]) == 3:
        print('length of conversation: ', df['utterance_events'].iloc[i])
        print('.................')
print(set_classifications)
