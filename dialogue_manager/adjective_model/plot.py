import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import torch
def get_compara_direction(cdict, embeddings, net, compara, source_str):
    comp_words = compara.split()
    if len(comp_words) == 1:
        emb1, emb2 = torch.zeros(300,), torch.from_numpy(embeddings[comp_words[0]])
    else:
        emb1, emb2 = torch.from_numpy(embeddings[comp_words[0]]), torch.from_numpy(embeddings[comp_words[1]])
    emb1, emb2 = torch.FloatTensor(emb1), torch.FloatTensor(emb2)
    source = cdict[source_str]
    wg = net(emb1, emb2, source)
    return wg.detach().numpy()

def plot_change(cdict, embeddings, net, compara, source_str, strength=1, save_path=None):
    ax = plt.gca()
    source = cdict[source_str].detach().numpy()
    direction = get_compara_direction(cdict, embeddings, net, compara, source_str)
    print(direction)
    ax = plt.gca()
    N = 100
    width, height = 1, 1
    for x in np.linspace(0, width, N):
        ax.add_patch(mpathes.Rectangle([x,0],width/N,height,color=np.clip(source+direction*x*strength,0,1)))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()