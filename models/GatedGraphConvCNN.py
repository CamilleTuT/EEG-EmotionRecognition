import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GatedGraphConv


class GatedGraphConvCNN(torch.nn.Module):
    def __init__(self, in_channels, n_graphs, hidden_channels, n_classes, dr):
        super(GatedGraphConvCNN, self).__init__()

        self.dr = dr

        first_hc = hidden_channels if hidden_channels > in_channels else in_channels

        self.gconv = GatedGraphConv(first_hc, 2)

        self.cnn1 = torch.nn.Conv1d(n_graphs, 1, kernel_size=3, stride=2)

        self.lin1 = torch.nn.Linear(32*(first_hc//2 - (1 if first_hc % 2 == 0 else 0)), hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, n_classes)

        self.act = nn.Softmax(dim=-1)

    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.gconv(x, edge_index, edge_attr)
        x = F.dropout(x, p=self.dr/2, training=self.training)
        x = x.relu()
        x = rearrange(x, '(bs g e) f -> (bs e) g f', bs=bs, e=32)
        x = self.cnn1(x).squeeze()
        x = x.relu()
        x = rearrange(x, '(bs e) f -> bs (e f)', bs=bs)
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = self.act(x)
        return x
