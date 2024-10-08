import torch
import os
import scipy
import skimage
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange
from torch.utils.data import Dataset
from npeet import entropy_estimators as ee


def data_1Dto2D_9_9(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], 0, data[16], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[1], 0, data[17], 0, 0, 0)
    data_2D[2] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[3] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[4] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[5] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[6] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[7] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:9*9
    return data_2D


def data_1Dto2D_8_9(data, Y=8, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, data[1], data[0], 0, data[16], data[17], 0, 0)
    data_2D[1] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[2] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[3] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[4] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[5] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[6] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[7] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:8*9
    return data_2D


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


class DEAPDataset_vision(Dataset):
    def __init__(self, root_dir, raw_dir, processed_dir, participant_from, participant_to):
        self.root = root_dir
        self.raw = raw_dir
        self.processed = processed_dir
        self.participant_from = participant_from
        self.participant_to = participant_to

        self.raw_dir = os.path.join(self.root, self.raw)
        self.processed_dir = os.path.join(self.root, self.processed)
        self.whether_processed = os.path.exists(processed_dir)

    @property
    def raw_file_names(self):
        raw_names = [f for f in os.listdir(self.raw_dir)]
        raw_names.sort()
        return raw_names

    def __getitem__(self, item):
        if self.whether_processed:
            return
        else:
            pbar = tqdm(range(self.participant_from, self.participant_to + 1))
            for participant_id in pbar:
                raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
                pbar.set_description(raw_name)
                participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
                baseline_data = torch.FloatTensor(participant_data['data'][:, :32, 128 * 2:128 * 3])
                signal_data = torch.FloatTensor(participant_data['data'][:, :32, 128 * 3:])
                for i, video in enumerate(signal_data[:self.n_videos, :, :]):
                    node_features = process_video(video, feature=self.feature)  # 160 * 60
                    baseline_feature = process_video(baseline_data[i, :, :], feature=self.feature)  # 160
                    baseline_feature = repeat(baseline_feature, 'a -> a b', b=60)
                    node_features = node_features - baseline_feature
                    node_features = torch.nn.functional.normalize(node_features)
            return

