import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from einops import rearrange


class GatedGraphConvMLP(torch.nn.Module):
    def __init__(self, in_channels, n_freq_bands, hidden_channels, n_classes, dr):
        super(GatedGraphConvMLP, self).__init__()

        self.dr = dr

        first_hc = hidden_channels if hidden_channels > in_channels else in_channels

        self.gconv = GatedGraphConv(first_hc, 2)

        self.lin1 = torch.nn.Linear(n_freq_bands*32*first_hc, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.lin3 = torch.nn.Linear(hidden_channels//2, hidden_channels//4)
        self.lin4 = torch.nn.Linear(hidden_channels//4, n_classes)

        self.act = nn.Softmax(dim=-1)

    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.gconv(x, edge_index, edge_attr)
        x = rearrange(x, '(bs g e) f -> bs (e g f)', bs=bs, e=32)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin4(x)
        x = self.act(x)
        return x
