import copy
import torch.optim as optim
from tqdm import tqdm
import torch.nn.init as init
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        init.xavier_normal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_normal_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
