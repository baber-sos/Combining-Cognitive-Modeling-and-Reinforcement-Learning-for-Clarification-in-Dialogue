import torch
import torch.nn as nn
class ColorNet_(nn.Module):
    def __init__(self):
        super(ColorNet_, self).__init__()
        self.fc1 = nn.Linear(603,30)
        self.fc2 = nn.Linear(33,3)
    def forward(self, emb1, emb2, source_color):
        x1 = self.fc1(torch.cat([emb1, emb2, source_color]))
        wg = self.fc2(torch.cat([x1,source_color]))
        return wg