from adjective_model.color_net_ import ColorNet_
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
def loss_(triples, cdict, embeddings):
    # with open("embeddings.pickle","rb") as f:
    #     embeddings = pickle.load(f)
    mse = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=0)
    my_loss = lambda source, target, wg: (1-cos(wg, target-source)) + mse(target, source+wg)


    epoches = 800
    net = ColorNet_()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for i in range(epoches):
        loss = 0
        for target_str, comp_words, source_str in triples:
            if len(comp_words) == 1:
                emb1, emb2 = torch.zeros(300,), torch.from_numpy(embeddings[comp_words[0]])
            else:
                emb1, emb2 = torch.from_numpy(embeddings[comp_words[0]]), torch.from_numpy(embeddings[comp_words[1]])
            emb1, emb2 = torch.FloatTensor(emb1), torch.FloatTensor(emb2)
            target, source = cdict[target_str], cdict[source_str]
            wg = net(emb1, emb2, source)
            loss += my_loss(source, target, wg)
        if i % 50 == 0:
            print(f"step:{i}, loss:{loss.detach().numpy()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net