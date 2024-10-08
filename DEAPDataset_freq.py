import os
import torch
import skimage
import pywt
import random
import scipy.io
import scipy.signal
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from einops import reduce, rearrange, repeat
from npeet import entropy_estimators as ee
from torch.optim.lr_scheduler import StepLR
from scipy.fft import rfft, rfftfreq, ifft
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from Electrodes import Electrodes
from tqdm import tqdm
import itertools


def train_val_test_split(dataset):
    train_mask = np.append(np.repeat(1, 30), np.repeat(0, 10))
    train_mask = np.tile(train_mask, int(len(dataset) / 40))
    val_mask = np.append(np.append(np.repeat(0, 30), np.repeat(1, 5)), np.repeat(0, 5))
    val_mask = np.tile(val_mask, int(len(dataset) / 40))
    test_mask = np.append(np.repeat(0, 35), np.repeat(1, 5))
    test_mask = np.tile(test_mask, int(len(dataset) / 40))

    train_set = [c for c in itertools.compress(dataset, train_mask)]
    val_set = [c for c in itertools.compress(dataset, val_mask)]
    test_set = [c for c in itertools.compress(dataset, test_mask)]

    return train_set, val_set, test_set


def calculate_de(window):
    return ee.entropy(window.reshape(-1, 1))


# Input: Video with shape (32,7680)
# Output: Graph node features with shape (5*32, 59) -> 5 graphs with 32 nodes each with 59 features each
def process_video(video, feature='psd'):
    # Transform to frequency domain
    fft_vals = np.fft.rfft(video, axis=-1)
    # Get frequencies for amplitudes in Hz
    samplingFrequency = 128
    fft_freq = np.fft.rfftfreq(video.shape[-1], 1.0 / samplingFrequency)
    # Delta, Theta, Alpha, Beta, Gamma
    bands = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    band_mask = np.array([np.logical_or(fft_freq < f, fft_freq > t) for f, t in bands])
    band_mask = repeat(band_mask, 'a b -> a c b', c=32)
    band_data = np.array(fft_vals)
    band_data = repeat(band_data, 'a b -> c a b', c=5)
    band_data[band_mask] = 0
    band_data = np.fft.irfft(band_data)
    windows = skimage.util.view_as_windows(band_data, (5, 32, 128), step=128).squeeze()
    # (5, 32, 60, 128)
    if windows.ndim == 4:  # video signal
        windows = rearrange(windows, 'a b c d -> b c a d')

        if feature == 'psd':
            features = scipy.signal.periodogram(windows)[1]
            features = np.mean(features, axis=-1)
        elif feature == 'de':
            features = np.apply_along_axis(calculate_de, -1, windows)
        features = rearrange(features, 'a b c -> (a b) c')
        features = torch.FloatTensor(features)
    else:  # baseline signal
        if feature == 'psd':
            features = scipy.signal.periodogram(windows)[1]
            features = np.mean(features, axis=-1)
        elif feature == 'de':
            features = np.apply_along_axis(calculate_de, -1, windows)
            features = np.expand_dims(features, axis=2)
        features = rearrange(features, 'a b c -> (a b) c').squeeze()
        features = torch.FloatTensor(features)

    return features


def plot_video(signal_data):
    electrodes = Electrodes()
    fig, axs = plt.subplots(32, sharex=True, figsize=(20, 50))
    fig.tight_layout()
    video = signal_data[0]
    for i in range(32):
        c = [float(i) / float(32), 0.0, float(32 - i) / float(32)]  # R,G,B
        axs[i].set_title(f'{electrodes.channel_names[i]}', loc='left', fontsize=20)
        axs[i].plot(video[i], color=c)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
    plt.savefig('eeg.png')


def remove_baseline_mean(signal_data):
    # Take first three senconds of data
    signal_baseline = np.array(signal_data[:, :, :128*3]).reshape(40, 32, 128, -1)
    # Mean of three senconds of baseline will be deducted from all windows
    signal_noise = np.mean(signal_baseline, axis=-1)
    # Expand mask
    signal_noise = repeat(signal_noise, 'a b c -> a b (d c)', d=60)
    return signal_data[:, :, 128*3:] - signal_noise


def process_video_wavelet(video, feature='energy', time_domain=False):
    band_widths = [32, 16, 8, 4]
    features = []
    for i in range(5):
        if i == 0:
            # Highest frequencies (64-128Hz) are not used
            cA, cD = pywt.dwt(video.numpy(), 'db4')
        else:
            cA, cD = pywt.dwt(cA, 'db4')

            cA_windows = skimage.util.view_as_windows(cA, (32, band_widths[i - 1] * 2), step=band_widths[i - 1]).squeeze()
            cA_windows = np.transpose(cA_windows[:59, :, :], (1, 0, 2))
            if feature == 'energy':
                cA_windows = np.square(cA_windows)
                cA_windows = np.sum(cA_windows, axis=-1)
                features.append(cA_windows)

    if time_domain:
        features = np.transpose(features, (2, 1, 0))
    features = rearrange(features, 'a b c -> (a b) c')
    features = torch.FloatTensor(features)

    # Normalization
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features -= m
    features /= s
    return features


class DEAPDatasetEEGFeatures(InMemoryDataset):
    def __init__(self, root, raw_dir, processed_dir, feature='de', transform=None, pre_transform=None,
                 include_edge_attr=True, undirected_graphs=True, add_global_connections=True, participant_from=1,
                 participant_to=32, n_videos=40):
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self.participant_from = participant_from
        self.participant_to = participant_to
        self.n_videos = n_videos
        self.feature = feature
        # Whether to include edge_attr in the dataset
        self.include_edge_attr = include_edge_attr
        # If true there will be 1024 links as opposed to 528
        self.undirected_graphs = undirected_graphs
        # Instantiate class to handle electrode positions
        print('Using global connections' if add_global_connections else 'Not using global connections')
        self.electrodes = Electrodes()
        super(DEAPDatasetEEGFeatures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        return [f'deap_processed_graph.{file_name}_{self.feature}.dataset']

    def process(self):
        # Number of nodes per graph
        n_nodes = len(self.electrodes.positions_3d)

        if self.undirected_graphs:
            source_nodes, target_nodes = np.repeat(np.arange(0, n_nodes), n_nodes), np.tile(np.arange(0, n_nodes), n_nodes)
        else:
            source_nodes, target_nodes = np.tril_indices(n_nodes, n_nodes)

        edge_attr = self.electrodes.adjacency_matrix[source_nodes, target_nodes]

        # Remove zero weight links
        mask = np.ma.masked_not_equal(edge_attr, 0).mask
        edge_attr, source_nodes, target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]

        edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # Expand edge_index and edge_attr to match windows
        e_edge_index = edge_index.clone()
        e_edge_attr = edge_attr.clone()
        number_of_graphs = 4
        for i in range(number_of_graphs - 1):
            a = edge_index + e_edge_index.max() + 1
            e_edge_index = torch.cat([e_edge_index, a], dim=1)
            e_edge_attr = torch.cat([e_edge_attr, edge_attr], dim=0)

        print(f'Number of graphs per video: {number_of_graphs}')
        # List of graphs that will be written to file
        data_list = []
        pbar = tqdm(range(self.participant_from, self.participant_to + 1))
        for participant_id in pbar:
            raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
            pbar.set_description(raw_name)
            # Load raw file as np array
            participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
            baseline_data = torch.FloatTensor(participant_data['data'][:, :32, 128*2:128*3])
            # signal_data = torch.FloatTensor(remove_baseline_mean(participant_data['data'][:, :32, :]))
            signal_data = torch.FloatTensor(participant_data['data'][:, :32, 128*3:])
            for i, video in enumerate(signal_data[:self.n_videos, :, :]):
                if self.feature == 'wav':
                    node_features = process_video_wavelet(video)
                else:
                    node_features = process_video(video, feature=self.feature)  # 160 * 60
                    baseline_feature = process_video(baseline_data[i, :, :], feature=self.feature)  # 160
                    baseline_feature = repeat(baseline_feature, 'a -> a b', b=60)
                    node_features = node_features - baseline_feature
                    node_features = F.normalize(node_features)

                if self.include_edge_attr:
                    data = Data(x=torch.FloatTensor(node_features), edge_attr=e_edge_attr, edge_index=e_edge_index, y=torch.FloatTensor([participant_data['labels'][i]]))
                else:
                    data = Data(x=torch.FloatTensor(node_features), edge_index=e_edge_index, y=torch.FloatTensor([participant_data['labels'][i]]))

                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
