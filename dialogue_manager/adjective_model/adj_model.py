import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import torch
import pickle
import dataset
# from adjective_model.Loss import loss
from adjective_model.color_net_ import ColorNet_

class AdjModel():
    def __init__(self):
        self.net = pickle.load(open(os.path.dirname(dataset.__file__) + '/adj_network.pkl', 'rb'))
        self.embeddings = pickle.load(open(os.path.dirname(dataset.__file__) + '/adj_embed.pkl', 'rb'))
        self.cdict = pickle.load(open(os.path.dirname(dataset.__file__) + '/color_dict.pkl', 'rb'))
        
    def get_compara_direction(self, compara, source_str):
        comp_words = compara.split()
        if len(comp_words) == 1:
            emb1, emb2 = torch.zeros(300,), torch.from_numpy(self.embeddings[comp_words[0]])
        else:
            emb1, emb2 = (torch.from_numpy(embeddings[comp_words[0]]), \
                torch.from_numpy(self.embeddings[comp_words[1]]))

        emb1, emb2 = torch.FloatTensor(emb1), torch.FloatTensor(emb2)
        source = None
        if type(source_str) == str:
            source = self.cdict[source_str.replace(' ', '')]
        else:
            source = torch.tensor(source_str, dtype=torch.float)

        wg = self.net(emb1, emb2, source)
        return wg.detach().numpy()
    
    def plot_change(self, compara, source_str, strength=1, save_path=None):
#         print(source_str)
        ax = plt.gca()
        source = None
        if type(source_str) == str:
            source = self.cdict[source_str.replace(' ', '')]
        else:
            source = torch.tensor(source_str, dtype=torch.float)
    #     source = source_str
        direction = self.get_compara_direction(compara, source_str)
#         print(direction)
        ax = plt.gca()
        N = 100
        width, height = 1, 1
        for x in np.linspace(0, width, N):
    #         print(x, width, N)
            ax.add_patch(mpathes.Rectangle([x,0],width/N,height,color=np.clip(source+direction*x*strength,0,1)))
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
# adj_model = AdjModel()