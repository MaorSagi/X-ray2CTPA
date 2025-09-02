import numpy as np

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
torch.backends.cudnn.benchmark = False

import torch.utils.data as data
from glob import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd
from params import *
import matplotlib.pyplot as plt
import numpy as np
# from scipy import signal
# from datetime import datetime
# import json
# import pandas as pd
import scipy
# import os
# # from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, roc_auc_score, \
# #     precision_recall_curve
# import torch
# import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# # from torchinfo import summary
# import unicodedata
# from tqdm import tqdm
# import numpy as np
# import scipy.signal
# from scipy import ndimage
# # import neurokit2 as nk
# from scipy.fftpack import dct, idct
# from torchvision import models
# import logging
#
# logger = logging.getLogger("my_log")
#
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
#
#
# class ECGDataset(BaseDataset):
#     """Dataset handler for ECG data."""
#
#     def __init__(self, list_IDs, data , labels, data_path, preload_device=None):
#         self.data = data
#         self.labels = labels
#         self.list_IDs = list_IDs
#         # should make the training quicker
#         self._preloaded = False
#         self.data_path = data_path
#         self._preloaded_dataset = {"x": [], "y": [], "IDs": []}
#         if preload_device:
#             for i in tqdm(range(len(self))):
#                 x, (y, IDs) = self[i]
#                 # self._preloaded_dataset['x'].append(x.to(preload_device, non_blocking=True))
#                 self._preloaded_dataset['x'] = pd.concat(
#                     [self._preloaded_dataset['x'], x.to(preload_device, non_blocking=True)])
#                 self._preloaded_dataset['y'] = pd.concat(
#                     [self._preloaded_dataset['y'], y.to(preload_device, non_blocking=True)])
#                 self._preloaded_dataset['IDs'] = pd.concat([self._preloaded_dataset['IDs'], IDs])
#             self._preloaded = True
#
#
#
#
#
#     def __getitem__(self, index):
#         'Generates one sample of data'
#         if self._preloaded:
#             return (self._preloaded_dataset['x'][index],
#                     (self._preloaded_dataset['y'][index],
#                      self._preloaded_dataset['IDs'][index]))
#         ID = self.list_IDs[index]
#         try:
#             X = np.load(self.data_path + str(ID) + '.npy')
#         except:
#             X = np.load(self.data_path + str(ID) + '.0.npy')
#
#         # X = X/1000 #Added to rescaling the signal
#         # X = self.apply_filter(X, **self.filters[0][1])
#         X, y = self.preprocess(X,ID)
#         return X, (y, ID)
#
#     def apply_filter(self, data, order, filter_type, freq, fs):
#         sos = scipy.signal.butter(order, freq, filter_type, output='sos', fs=fs)
#         return scipy.signal.sosfiltfilt(sos, data.T).T
#
#     def apply_baselinewander(self, data):
#         filter_data = np.zeros_like(data)
#         for i in range(12):
#             # filter_data[:,i]=scipy.signal.medfilt(data[:,i], 201)
#             filter_data[:, i] = ndimage.median_filter(data[:, i], 201)
#
#             # logger.info(filter_data.shape)
#         # return ndimage.median_filter(data, size=201, axes=(0,))
#         return filter_data
#
#     def apply_shift(self, data, shift_range=[-5, 5], shift_multiplier=1):
#         rand_shift = np.random.randint(shift_range[0], shift_range[1])
#         shifted_data = np.roll(data, shift_multiplier * rand_shift, axis=0)
#
#         return shifted_data
#
#     def apply_permutation(self, data, fs=500):
#         _, r_peaks = nk.ecg_peaks(data[:, 1], fs, method='pantompkins1985')
#         r_peak_indices = r_peaks['ECG_R_Peaks']
#
#         rr_segments = np.split(data, r_peak_indices)
#
#         np.random.shuffle(rr_segments)
#         permuted_ecg = np.concatenate(rr_segments)
#         return permuted_ecg
#
#     def apply_HR_modulation(self, data, n_samples=5000, curr_fs=500,
#                             max_bpm_multiplier=3):
#         new_fs = np.random.randint(-4, 5) * 100 + curr_fs
#         resampled_data = np.zeros((n_samples, data.shape[1]))
#         new_num_samples = int(data.shape[0] * new_fs / curr_fs)
#         resampled_data = scipy.signal.resample(data, new_num_samples, axis=0)
#
#         if len(resampled_data) > n_samples:
#
#             start_idx = np.random.randint(0, len(resampled_data) - n_samples)
#             resampled_data = resampled_data[start_idx: start_idx + n_samples]
#
#         elif len(resampled_data) < n_samples:
#
#             bpm_multiplier = np.random.randint(1, max_bpm_multiplier)
#
#             ecg_extended = np.tile(data, (int(bpm_multiplier), 1))
#
#             mod = len(ecg_extended) % bpm_multiplier
#
#             ecg_extended = np.concatenate([ecg_extended, data[:mod]], axis=0)
#
#             resampled_data = nk.signal_resample(ecg_extended, desired_length=n_samples, method='fft')
#         return resampled_data
#
#     def gaussian_noise(self, data, snr=0.01):
#         noise_power = data.var() * snr
#
#         noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
#
#         noisy_ecg = data + noise
#
#         return noisy_ecg
#
#     def add_freq_dropout(self, data, dropout_rate=0.1):
#         ecg_dct = dct(data, type=2, norm='ortho', axis=0)
#
#         num_samples, num_leads = ecg_dct.shape
#
#         num_components_to_drop = int(dropout_rate * num_samples)
#         dropout_indices = np.random.choice(np.arange(num_samples), size=num_components_to_drop, replace=False)
#
#         ecg_dct[dropout_indices, :] = 0
#
#         ecg_idct = idct(ecg_dct, type=2, norm='ortho', axis=0)
#
#         return ecg_idct
#
#     def add_mixed_sine_waves(self, data, num_segments=5, freq_range=(0.1, 1.0), amplitude_factor=0.1):
#
#         augmented_ecg = data.astype(np.float64).copy()
#
#         for segment_idx in range(num_segments):
#             start_idx = np.random.randint(0, len(data))
#
#             lead = np.random.randint(0, data.shape[-1])
#
#             freq = np.random.uniform(freq_range[0], freq_range[1])
#
#             t = np.linspace(0, 1 / 2, int(250 / freq), endpoint=False)
#
#             end_idx = min(len(data), start_idx + len(t))
#
#             t = t[:end_idx - start_idx]
#
#             max_amplitude = np.max(np.abs(data[:, lead]))
#
#             sine_wave = np.random.choice([-1, 1]) * amplitude_factor * max_amplitude * np.sin(2 * np.pi * t)
#
#             augmented_ecg[start_idx:end_idx, lead] += sine_wave
#
#         return augmented_ecg

class ECGXrayCTPADataset(data.Dataset):
    def __init__(self, root='.', target=None, mode="train", augmentation=False):
        if target is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))


        self.mode = mode
        self.root = root
        self.cts = self.root + 'preprocessed_data/sheba_encryp/ctpa/'
        self.xrays = self.root + 'preprocessed_data/sheba_encryp/xray/'
        self.ecgs = self.root + 'data/ecg/'
        if target is not None:
            self.data = pd.read_csv(root + target)
            cts_list = [ct.replace(".npy","") for ct in os.listdir(self.cts)]
            xrays_list = [xr.replace(".npy","") for xr in os.listdir(self.xrays)]
            ecgs_list = [ecg.replace(".npy","") for ecg in os.listdir(self.ecgs)]
            self.data[CT_ACCESSION_COL] = self.data[CT_ACCESSION_COL].astype(str).apply(lambda s: s.replace(".0",""))
            self.data = self.data[self.data[CT_ACCESSION_COL].isin(cts_list)]
            self.data[XRAY_ACCESSION_COL] = self.data[XRAY_ACCESSION_COL].astype(float).astype(str).apply(lambda s: s.replace(".0",""))
            self.data = self.data[self.data[XRAY_ACCESSION_COL].isin(xrays_list)]
            self.data[ECG_ACCESSION_COL] = self.data[ECG_ACCESSION_COL].astype(str).replace(".0","")
            self.data = self.data[self.data[ECG_ACCESSION_COL].isin(ecgs_list)]
            self.data = self.data.reset_index()
        self.augmentation = augmentation
        self.VAE = True
        self.text_label = False

    def __len__(self):
        return len(self.data)

    def preprocess_ecg(self, X, ID):
        """Preprocess the raw data."""
        if X.shape[0] != 1:
            X = np.expand_dims(X, axis=0)
        if X.shape[1] == 2500:
            # logger.info('will upsampling to 5000')
            X = np.squeeze(X, axis=0)
            X = scipy.signal.resample(X, 5000)
            X = np.expand_dims(X, axis=0)
            X = torch.from_numpy(X)
        elif X.shape[1] == 5000:
            # X = np.squeeze(X, axis=0)  # relevant to ECG data
            # X = np.expand_dims(X, axis=2)  # relevant to ECG data
            X = torch.from_numpy(X)
        else:
            raise ValueError(f'Note size is invalid, this sample number: {ID}')

        X = X.type(torch.FloatTensor)
        return X

    def __getitem__(self, idx):
        ct_accession = self.data.loc[idx, CT_ACCESSION_COL]
        cxr_accession = self.data.loc[idx, XRAY_ACCESSION_COL]
        ecg_accession = self.data.loc[idx, ECG_ACCESSION_COL]
        label = self.data.loc[idx, LABEL_COL]
        label = F.one_hot(torch.tensor(label).to(torch.int64), num_classes=2).to(torch.float32)
        if not self.text_label:
            label = torch.tensor(label).type(torch.DoubleTensor)
            # label = label.reshape(1)
        else:
            if label:
                label = "Positive"
            else:
                label = "Negative"
        # Load the CTPA 3D scan
        ct =  np.load(self.cts+ str(ct_accession) + '.npy').astype(np.float32)

        if not self.VAE:
            if self.augmentation:
                random_n = torch.rand(1)
                if random_n[0] > 0.5:
                    ct = np.flip(ct, 0)

        ctout = torch.from_numpy(ct.copy()).float()
        if not self.VAE:
            ctout = ctout.unsqueeze(0)
        else:
            ctout = ctout.permute(0,3,1,2)

        # Load matching Xray 2D image
        xray = torch.from_numpy(np.load(self.xrays + str(cxr_accession) + '.npy')).float()

        # Load matching ECG
        ecg = torch.from_numpy(np.load(self.ecgs + str(ecg_accession) + '.npy')).float()
        ecg = self.preprocess_ecg(ecg, ecg_accession)

        if self.mode == "train" or self.mode == "test":
            return {'ct': ctout, 'cxr': xray, 'ecg':ecg, 'target': label}
        else: #if self.mode == "infer"
            return {'ct': ctout, 'cxr': xray, 'ecg':ecg, 'ct_accession': ct_accession, 'cxr_accession': cxr_accession, 'ecg_accession': ecg_accession}
