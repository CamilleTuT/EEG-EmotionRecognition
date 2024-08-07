import os
import torch
import scipy
import scipy.io
import itertools
import matplotlib
import matplotlib.pyplot as plt
import mne
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Electrodes3 import Electrodes
from einops import rearrange
import pickle
import pandas as pd
import numpy as np
import pyeeg as pe


# Get 30 videos for each participant for test, 5 for validation and 5 for testing
def train_val_test_split(dataset):
    train_mask = np.append(np.repeat(1, 30), np.repeat(0, 10))
    # print(train_mask)
    train_mask = np.tile(train_mask, int(len(dataset) / 40))
    # print(train_mask)
    val_mask = np.append(np.append(np.repeat(0, 30), np.repeat(1, 5)), np.repeat(0, 5))
    val_mask = np.tile(val_mask, int(len(dataset) / 40))
    test_mask = np.append(np.repeat(0, 35), np.repeat(1, 5))
    test_mask = np.tile(test_mask, int(len(dataset) / 40))

    train_set = [c for c in itertools.compress(dataset, train_mask)]
    # print(type(train_set))
    # print(len(train_set))
    # print(train_set)
    val_set = [c for c in itertools.compress(dataset, val_mask)]
    test_set = [c for c in itertools.compress(dataset, test_mask)]

    return train_set, val_set, test_set


def plot_graph(graph_data):
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    from matplotlib import pyplot as plt
    graph = to_networkx(graph_data)

    plt.figure(1, figsize=(7, 6))
    nx.draw(graph, cmap=plt.get_cmap('Set1'), node_size=75, linewidths=6)
    plt.show()


def add_channel_names(ax, pos, names):
    for name, (x, y) in zip(names, pos):
        ax.text(x, y, name, color='black', ha='center', va='center')


def visualize_window(window):
    window = window.cpu().detach().numpy()[:12]

    eeg_mean = window.mean(axis=-1)
    chunks = eeg_mean.T
    electrodes = Electrodes()
    # Show all chunks and label (12)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    for i, chunk in enumerate(eeg_mean):
        index = np.unravel_index(i, (3, 4))
        ax = axes[index[0]][index[1]]
        ax.title.set_text(f'Chunk number {i} (seconds {5.25 * i} to {5.25 * (i + 1)})')
        im, _ = mne.viz.plot_topomap(chunk, electrodes.positions_2d, names=electrodes.channel_names, axes=ax,
                                     cmap='bwr ', show=False)
        add_channel_names(ax, electrodes.positions_2d, electrodes.channel_names)
    plt.show()


def visualize_graph(graph_features):
    print(graph_features.shape)
    electrodes = Electrodes()
    eeg_mean = graph_features.cpu().detach().numpy().mean(axis=-1)
    # Show video as 1 chunk
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(eeg_mean, electrodes.positions_2d, names=electrodes.channel_names, axes=ax, cmap='bwr',
                                 show=False)
    add_channel_names(ax, electrodes.positions_2d, electrodes.channel_names)
    plt.show()


def describe_graph(graph_data):
    print(graph_data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {graph_data.num_nodes}')
    print(f'Number of edges: {graph_data.num_edges}')
    print(f'Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {graph_data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {graph_data.contains_self_loops()}')
    print(f'Is undirected: {graph_data.is_undirected()}')


class DEAPDataset(InMemoryDataset):
    # 1 participant per dataset
    # Theoretically it doesn't make sense to train for all participants -> unless aiming for subject-independent classification (not atm)
    # PyG represents graphs sparsely, which refers to only holding the coordinates/values for which entries in  A  are non-zero.
    def __init__(self, root, raw_dir, processed_dir, participant_from, participant_to=None, include_edge_attr=True,
                 undirected_graphs=True, transform=None, pre_transform=None, window_size=None):
        self._raw_dir = raw_dir
        self.i = 1
        self._processed_dir = processed_dir
        self.participant_from = participant_from
        self.participant_to = participant_from if participant_to is None else participant_to
        # Whether to include edge_attr in the dataset
        self.include_edge_attr = include_edge_attr
        # If true there will be 1024 links as opposed to 528
        self.undirected_graphs = undirected_graphs
        # Instantiate class to handle electrode positions
        self.electrodes = Electrodes()
        # Define the size of the windows -> 672: 12, 5.25 second windows
        self.window_size = window_size
        # print(self.i)
        # print('process' in self.__class__.__dict__.keys())
        super(DEAPDataset, self).__init__(root, transform, pre_transform)
        # print('process' in self.__class__.__dict__.keys())
        # self.process()
        # print(self.i)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print(self.data)
        # print(type(self.data))
        # print(type(self.slices))
        # print(self.processed_paths[0])

    @property
    def raw_dir(self):
        return f'{self.root}/{self._raw_dir}'

    @property
    def processed_dir(self):
        return f'{self.root}/{self._processed_dir}'

    @property
    def raw_file_names(self):
        raw_names = [f for f in os.listdir(self.raw_dir)]
        raw_names.sort()
        return raw_names

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        file_name = f'{self.participant_from}-{self.participant_to}' if self.participant_from is not self.participant_to else f'{self.participant_from}'
        # print(f'deap_processed_graph.{file_name}.dataset')
        return [f'deap_processed_graph.{file_name}.dataset']

    def get_adj(self):
        pbar = tqdm(range(self.participant_from, 2))
        for participant_id in pbar:
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            # pbar.set_description(raw_name)
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            signal_data = torch.FloatTensor(participant_data['data'][:, :32, 384:])
            for index_video, node_features in enumerate(signal_data):
                adj = np.zeros((32, 32))
                j = 0
                while j < 32:
                    k = 0
                    while k < 32:
                        # print(node_features[j].shape)
                        # print(node_features[k].shape)
                        ad = np.corrcoef(node_features[j], node_features[k])
                        adj[j][k] = ad[0][1]
                        k = k + 1
                    j = j + 1
                print('2')
                local_conn_mask = abs(adj) > 0.5
                print('3')
                adj = adj * local_conn_mask
                return adj

    def process(self):
        # Number of nodes per graph
        self.i = 2
        n_nodes = len(self.electrodes.channel_names)

        NODE_FEATURE_N = 8064
        if self.window_size is not None and NODE_FEATURE_N % self.window_size != 0:
            raise 'Error, window number of features should be divisible by window size'

        if self.undirected_graphs:
            source_nodes, target_nodes = np.repeat(np.arange(0, n_nodes), n_nodes), np.tile(np.arange(0, n_nodes), n_nodes)
        else:
            source_nodes, target_nodes = np.tril_indices(n_nodes, n_nodes)
        # print(source_nodes)
        # print(source_nodes.shape)
        # print(target_nodes)
        # print(target_nodes.shape)
        # print('1')
        # edge_attr = self.get_adj()
        # print(edge_attr)
        # edge_attr=edge_attr[source_nodes,target_nodes]
        # edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
        # print(edge_attr)
        # print(edge_attr.shape)
        # Remove zero weight links
        # mask = np.ma.masked_not_equal(edge_attr, 0).mask
        # edge_attr,source_nodes,target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]
        # print(edge_attr)
        # print(edge_attr.shape)
        # print(source_nodes)
        # print(source_nodes.shape)
        # print(target_nodes)
        # print(target_nodes.shape)
        # edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes,target_nodes], dtype=torch.long)

        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from, self.participant_to + 1))
        for participant_id in pbar:
            # print(participant_id)
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            # print(raw_name)
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            # print('001')
            signal_data = torch.FloatTensor(participant_data['data'][:, :32, 384:])
            labels = torch.Tensor(participant_data['labels'])
            # Create time windows
            if self.window_size != None:
                signal_data = rearrange(signal_data, 'v c (s w) -> v s c w', w=self.window_size)
            # print('002')
            # Enumerate videos / graphs ->
            for index_video, node_features in enumerate(signal_data):
                # Create graph
                # print(index_video)
                # print(index_video.shape)
                # print(node_features)
                # print(node_features.shape)
                # print(labels[index_video])
                # print(labels[index_video].shape)
                adj = np.zeros((32, 32))
                j = 0
                while j < 32:
                    k = 0
                    while k < 32:
                        # print(node_features[j].shape)
                        # print(node_features[k].shape)
                        ad = np.corrcoef(node_features[j], node_features[k])
                        adj[j][k] = ad[0][1]
                        k = k + 1
                    j = j + 1
                print('2')
                local_conn_mask = abs(adj) > 0.5
                print('3')
                edge_attr = adj * local_conn_mask
                edge_attr = edge_attr[source_nodes, target_nodes]
                mask = np.ma.masked_not_equal(edge_attr, 0).mask
                edge_attr, source_nodes1, target_nodes1 = edge_attr[mask], source_nodes[mask], target_nodes[mask]
                edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes1, target_nodes1],
                                                                                   dtype=torch.long)
                y = torch.FloatTensor(labels[index_video]).unsqueeze(0)
                # print(y)
                # print(y.shape)
                # 1 graph per window (12 per video with window size 672)
                data = Data(x=node_features, edge_attr=edge_attr, edge_index=edge_index,
                            y=y) if self.include_edge_attr else Data(x=node_features, edge_index=edge_index, y=y)
                data_list.append(data)
                # print('4')
        data, slices = self.collate(data_list)
        # print('5')
        # print(slices)
        torch.save((data, slices), self.processed_paths[0])


def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
    meta = []
    file_path = r'D:/roo/pyt/data_preprocessed_python'
    with open(file_path + '\s' + sub + '.dat', 'rb') as file:

        subject = pickle.load(file, encoding='latin1')  # resolve the python 2 data problem by encoding : latin1

        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0

            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channel:
                    X = data[j][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec以0.125秒为间隔，在2秒内对原始数据进行切片
                    Y = pe.bin_power(X, band, sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    meta_data = meta_data + list(Y[0])

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start = start + step_size

        meta = np.array(meta)
        np.save('out' + sub, meta, allow_pickle=True, fix_imports=True)


def calc_features(data):
    result = []
    result.append(np.mean(data))
    result.append(np.median(data))
    result.append(np.max(data))
    result.append(np.min(data))
    result.append(np.std(data))
    result.append(np.var(data))
    result.append(np.max(data) - np.min(data))
    result.append(pd.Series(data).skew())
    result.append(pd.Series(data).kurt())
    return result
