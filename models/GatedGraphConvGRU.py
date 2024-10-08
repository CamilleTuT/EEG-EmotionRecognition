import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GatedGraphConv
import torch.nn.utils.weight_norm as weight_norm


class GatedGraphConvGRU(torch.nn.Module):
    def __init__(self, in_channels, n_freq_bands, hidden_channels, n_classes, dr, target, training):
        super(GatedGraphConvGRU, self).__init__()

        self.dr = dr
        self.hidden_channels = hidden_channels
        self.grus = nn.ModuleList([nn.GRU(32, hidden_channels, 2, batch_first=True) for _ in range(n_freq_bands)])
        self.gconv = GatedGraphConv(hidden_channels if hidden_channels > in_channels else in_channels, 2)
        self.lin1 = weight_norm(nn.Linear(n_freq_bands*hidden_channels, hidden_channels))
        self.lin2 = nn.Linear(hidden_channels, n_classes)
        self.act = nn.Softmax(dim=-1)
        self.act1 = nn.Sigmoid()

        self.target = {'valence': 0, 'arousal': 1, 'dominance': 2, 'liking': 3}[target]
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.eval_losses = []
        self.training = training

    def forward(self, batch):
        bs = len(torch.unique(batch.batch)) if 'batch' in dir(batch) else 1
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.gconv(x, edge_index, edge_attr)

        x = rearrange(x, '(bs g e) f -> g bs f e', bs=bs, e=32)

        xs = []
        for i, freq_band_x in enumerate(x):
            out, h_n = self.grus[i](freq_band_x)
            xs.append(h_n[-1])

        x = torch.stack(xs)
        x = rearrange(x, 'g bs hc -> bs (g hc)')
        x = F.dropout(x, p=self.dr/2, training=self.training)

        x = self.lin1(x)
        x = x.selu()
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.lin2(x)
        x = self.act1(x)
        return x
