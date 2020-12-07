import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, obs_params, outputs, num_layers):
        super(DQN, self).__init__()
        self.net_layers = []
        linear_input_size = 1
        for i in obs_params:
            linear_input_size *= i
        output_size = linear_input_size
        for i in range(num_layers):
            if i == num_layers - 1:
                output_size = outputs
            self.net_layers.append(nn.Linear(linear_input_size, output_size))

        self.net_layers = nn.ModuleList(self.net_layers)
    
    def forward(self, x):
        for i in range(len(self.net_layers) - 1):
            x = F.relu(self.net_layers[i](x))
        return self.net_layers[-1](x)
