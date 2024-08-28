import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GraphConv
from einops import rearrange
# from DEAPDataset_Spacial import visualize_graph


class GNNLSTM(torch.nn.Module):
    def __init__(self, target, training):
        super(GNNLSTM, self).__init__()

        self.gconv1 = GraphConv(in_channels=7680, out_channels=5000, aggr='add')
        self.gconv2 = GraphConv(in_channels=5000, out_channels=4000, aggr='add')
        self.lstm = torch.nn.LSTM(2, 3, 2, bidirectional=True)
        self.mlp = Sequential(Linear(15000, 1))
        # MODEL CLASS ATTRIBUTES
        self.target = {'valence': 0, 'arousal': 1, 'dominance': 2, 'liking': 3}[target]
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.eval_losses = []
        self.training = training

    def forward(self, batch, visualize_convolutions=False):
        x = batch.x
        print(x.shape)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch = batch.batch
        # Information propagation trough graph visualization
        x = self.gconv1(x, edge_index, edge_attr)  # torch.Size([2048, 5000])
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # x = self.gconv2(x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)  # torch.Size([2048, 4000])
        x = global_add_pool(x, batch)  # torch.Size([64, 5000])
        x = rearrange(x, 'b (sl i) -> sl b i', i=2)  # torch.Size([2500, 64, 2])
        output, (c_n, h_n) = self.lstm(x)  # torch.Size([2500, 64, 6])
        x = rearrange(output, 'sl b i -> b (sl i)')  # torch.Size([64, 15000])
        x = self.mlp(x)
        return x
