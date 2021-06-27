import math
import torch
import torch.nn as nn
from torch.distributions import Independent
from torch.distributions import Normal
from torch.distributions import Uniform


class RandomPolicy(nn.Module):
    def __init__(self, n_traj, action_dim, uniform=False):
        super(RandomPolicy, self).__init__()
        self.output_dim = n_traj
        self.input_len = action_dim
        self.uniform = uniform

        if self.uniform:
            self.random_policy = Uniform(-1.0, 1.0).expand(torch.Size([self.output_dim, self.input_len]))
        else:
            self.random_policy = Independent(Normal(
                loc=torch.zeros(self.output_dim, self.input_len),
                scale=0.5 * torch.ones(self.output_dim, self.input_len)), 1
            )

    def forward(self, x):
        return self.random_policy


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50, append=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.append = append
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(1, d_model+1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.append:
            x = torch.cat((x,self.pe[:x.size(0), :].repeat(1,x.size(1),1)),2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)