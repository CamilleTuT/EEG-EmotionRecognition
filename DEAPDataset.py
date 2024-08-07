import os
import torch
import scipy
import scipy.io
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import mne
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from tqdm import tqdm
from Electrodes3 import Electrodes
from einops import rearrange


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
    # 得到了三个掩码：train_mask用于训练集、val_mask用于验证集、test_mask用于测试集。
    # 这些掩码可以用来从原始数据集中选择出相应的样本用于训练、验证和测试。

    train_set = [c for c in itertools.compress(dataset, train_mask)]
    # print(type(train_set))
    # print(len(train_set))
    # print(train_set)
    val_set = [c for c in itertools.compress(dataset, val_mask)]
    test_set = [c for c in itertools.compress(dataset, test_mask)]
    # 据不同的掩码将数据集划分为训练集、验证集和测试集，并将其分别存储在 train_set、val_set 和 test_set 中。
    return train_set, val_set, test_set


def plot_graph(graph_data):
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    from matplotlib import pyplot as plt
    graph = to_networkx(graph_data)

    plt.figure(1, figsize=(7, 6))
    nx.draw(graph, cmap=plt.get_cmap('Set1'), node_size=75, linewidths=6)  # 节点大小为 75，边的宽度为 6。
    plt.show()
    # 绘制图形


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
        im, _ = mne.viz.topomap.plot_topomap(chunk, electrodes.positions_2d, names=electrodes.channel_names,
                                             show_names=True, axes=ax, cmap='bwr ', show=False)
    plt.show()


def visualize_graph(graph_features):
    print(graph_features.shape)
    electrodes = Electrodes()
    eeg_mean = graph_features.cpu().detach().numpy().mean(axis=-1)
    # Show video as 1 chunk
    fig, ax = plt.subplots()
    im, _ = mne.viz.topomap.plot_topomap(eeg_mean, electrodes.positions_2d, names=electrodes.channel_names,
                                         show_names=True, axes=ax, cmap='bwr', show=False)
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
        # Whether or not to include edge_attr in the dataset
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
            pbar.set_description(raw_name)
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            signal_data = torch.FloatTensor(participant_data['data'][:, :32, 384:])
            # 这个切片操作选择了从第384个时间点开始的所有数据。通常情况下，原始脑电图（EEG）数据是一个三维数组，其维度可能是(时间点数, 通道数, 特征数)
            # 这里的384表示数据从第384个时间点开始，选取所有通道（32个通道），而:则表示选取所有的特征数。
            for index_video, node_features in enumerate(signal_data):
                adj = np.zeros((32, 32))  # 这行代码创建了一个 32x32 大小的零矩阵，用于存储邻接矩阵。
                j = 0
                while j < 32:
                    k = 0
                    while k < 32:
                        # print(node_features[j].shape)
                        # print(node_features[k].shape)
                        ad = np.corrcoef(node_features[j], node_features[
                            k])  # 这行代码使用 numpy.corrcoef 函数计算了节点 j 和节点 k 之间的相关系数，并将结果存储在 ad 变量中。
                        adj[j][k] = ad[0][1]  # 这行代码将计算得到的相关系数存储到邻接矩阵中的对应位置。
                        k = k + 1
                    j = j + 1
                print('2')
                local_conn_mask = abs(adj) > 0.5  # 这行代码创建了一个局部连接掩码，用于保留大于 0.5 的相关系数。
                print('3')
                adj = adj * local_conn_mask  # 这行代码将邻接矩阵与局部连接掩码相乘，以实现对小于 0.5 的相关系数的剔除。
                return adj
                # 计算图的邻接矩阵

    def process(self):
        # Number of nodes per graph
        self.i = 2
        n_nodes = len(self.electrodes.channel_names)  # 确定图中节点的数量

        NODE_FEATURE_N = 8064
        if self.window_size is not None and NODE_FEATURE_N % self.window_size != 0:
            raise 'Error, window number of features should be divisible by window size'

        if self.undirected_graphs:
            source_nodes, target_nodes = np.repeat(np.arange(0, n_nodes), n_nodes), np.tile(np.arange(0, n_nodes),
                                                                                            n_nodes)
        else:
            source_nodes, target_nodes = np.tril_indices(n_nodes,
                                                         n_nodes)  # 如果要求无向图，则每个节点与所有节点相连；否则，只取下三角部分的索引，避免重复添加边。
        # print(source_nodes)
        # print(source_nodes.shape)
        # print(target_nodes)
        # print(target_nodes.shape)
        print('1')
        edge_attr = self.get_adj()
        print(edge_attr)
        edge_attr = edge_attr[
            source_nodes, target_nodes]  # source_nodes 包含了所有边的源节点索引，target_nodes 包含了所有边的目标节点索引，那么 edge_attr[source_nodes, target_nodes] 就会返回一个数组，其中每个元素都对应于一条边的属性值。
        # edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
        # print(edge_attr)
        # print(edge_attr.shape)
        # Remove zero weight links
        mask = np.ma.masked_not_equal(edge_attr, 0).mask  # 移除权重为零的边。
        edge_attr, source_nodes, target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[
            mask]  # 是根据 mask 中的布尔值，筛选出满足条件的边的属性、源节点索引和目标节点索引，并将它们重新赋值给 edge_attr、source_nodes、target_nodes。这样做的结果是，剔除了 mask 中对应位置为 False 的元素，保留了 mask 中对应位置为 True 的元素，从而得到了筛选后的边的属性、源节点索引和目标节点索引。
        # print(edge_attr)
        # print(edge_attr.shape)
        # print(source_nodes)
        # print(source_nodes.shape)
        # print(target_nodes)
        # print(target_nodes.shape)
        edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes, target_nodes],
                                                                           dtype=torch.long)  # 创建 PyTorch 中的张量表示图的边，并存储到 edge_attr 和 edge_index 中
        # torch.FloatTensor(edge_attr): 这部分将 NumPy 数组 edge_attr 转换为 PyTorch 的 FloatTensor 类型。edge_attr 是边的属性值，可能是一些浮点数或者其他数值类型，转换为 FloatTensor 后可以在 PyTorch 中进行处理。
        # torch.tensor([source_nodes,target_nodes], dtype=torch.long): 这部分创建了一个张量 edge_index，其中包含了所有边的源节点索引和目标节点索引。source_nodes 和 target_nodes 是两个一维数组，分别表示每条边的源节点索引和目标节点索引。在创建张量时，这两个数组被组合成一个二维数组，其中第一行是所有的源节点索引，第二行是所有的目标节点索引。dtype=torch.long 指定了张量的数据类型为长整型，因为节点索引通常是整数类型。
        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from, self.participant_to + 1))
        for participant_id in pbar:
            # print(participant_id)
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            # print(raw_name)
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')  # 读取mat文件
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
# 将原始数据处理成适合用于图神经网络训练的数据格式，并保存到文件中。
