import torch.nn as nn
import torch

from activation import Activation

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, pathway_info, act_func=None,
                 n_layers=4):
        super(Net, self).__init__()
        self.hidden_dim = pathway_info.shape[1]
        self.n_layers = n_layers
        self.fc1 = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.pathway_info = pathway_info.float()
        self.activation = Activation(act_func)

        self.hidden = nn.ModuleList()
        in_dim = self.hidden_dim
        for _ in range(n_layers - 2):
            self.hidden.append(nn.Linear(in_dim, 64, bias=False))
            in_dim = 64

        self.fc_out = nn.Linear(in_dim, output_dim, bias=False)

    def forward(self, x):
        pathway = torch.matmul(x, (self.pathway_info * self.fc1.weight.t()))
        h = self.activation(pathway)
        for layer in self.hidden:
            h = self.activation(layer(h))
        return self.fc_out(h)

def make_init_fn():
    def init_fn(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
    return init_fn
