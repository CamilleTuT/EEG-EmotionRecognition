import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv
from einops import reduce, rearrange
import torch.nn.init as init
# from layers import GCN, HGPSLPoo
from DEAPDataset2 import visualize_graph
import torch.optim as optim
from tqdm import tqdm
import copy
from MLPinit import MLP


class GatedAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedAttention, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)
        self.U = nn.Linear(in_channels, out_channels)
        self.V = nn.Linear(out_channels, 1)

        self.query_linear = nn.Linear(in_channels, out_channels)
        self.key_linear = nn.Linear(in_channels, out_channels)
        self.value_linear = nn.Linear(in_channels, out_channels)

    def forward(self, z, edge_index):
        h = torch.tanh(self.W(z))
        u = torch.sigmoid(self.U(z))
        alpha = self.V(h * u)
        alpha = torch.softmax(alpha, dim=0)

        query = self.query_linear(z)
        key = self.key_linear(z)
        value = self.value_linear(z)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = torch.softmax(scores, dim=1)
        x = value * alpha
        return x, attention_weights


class GNNLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, target, num_layers=2):
        super(GNNLSTM, self).__init__()

        self.gconv1 = GraphConv(in_channels=7680, out_channels=5000, aggr='add')
        self.gconv2 = GraphConv(in_channels=5000, out_channels=4000, aggr='add')

        self.gated_attention = GatedAttention(in_channels=5000, out_channels=5000)
        self.lstm = nn.LSTM(2, 3, 2, bidirectional=True)
        #self.mlp = Sequential(Linear(15000, 16),ReLU(), Linear(16, 1,),ReLU())
        self.mlp = Sequential(Linear(15000, 1))
        # MODEL CLASS ATTRIBUTES
        self.target = {'valence': 0, 'arousal': 1, 'dominance': 2, 'liking': 3}[target]
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.eval_losses = []
        self.eval_patience_count = 0
        self.eval_patience_reached = False
        self.train_acc = 0
        self.eval_acc = 0

    def forward(self, batch, visualize_convolutions=False):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch = batch.batch
        bs = len(torch.unique(batch))
        # print(batch)
        # print(torch.unique(batch))
        # print(bs)
        # Information propagation trough graph visualization
        if visualize_convolutions:
            visualize_graph(x[:32])

        x = self.gconv1(x, edge_index, edge_attr)
        # print(x.shape)
        # x=torch.tanh(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.shape)
        if visualize_convolutions:
            visualize_graph(x[:32])

        # x = self.gconv2(x,edge_index,edge_attr)
        # x = F.dropout(x, p=0.2, training=self.training)
        x, attention_weights = self.gated_attention(x, edge_index)
        # if visualize_convolutions:
        #   visualize_graph(x[:32])
        x = gaddp(x, batch)
        # print(x.shape)

        x = rearrange(x, 'b (sl i) -> sl b i', i=2)
        # print(x.shape)
        output, (c_n, h_n) = self.lstm(x)
        # print(output.shape)
        x = rearrange(output, 'sl b i -> b (sl i)')
        # print(x.shape)
        x = self.mlp(x)
        # print(x.shape)
        return x

    def GNN_Init(data, model, epochs=299):
        best_model_wts = None
        best_loss = float('inf')
        mlp = MLP(input_dim=data.x.shape[1], hidden_dim=256, output_dim=data.y.max().item() + 1).cuda()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001, weight_decay=5e-7)
        criterion = nn.CrossEntropyLoss().cuda()
        for _ in tqdm(range(epochs)):
            mlp.train()
            optimizer.zero_grad()
            out = mlp(data.x)
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            optimizer.step()
            mlp.eval()
            val_out = mlp(data.x)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_model_wts = copy.deepcopy(mlp.state_dict())

        model.gconv1.lin.weight.data = best_model_wts['fc1.weight']
        model.gconv1.bias.data = best_model_wts['fc1.bias']
        model.gconv2.lin.weight.data = best_model_wts['fc2.weight']
        model.gconv2.bias.data = best_model_wts['fc2.bias']
        return model
