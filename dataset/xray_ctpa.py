import numpy as np

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F

import torch.utils.data as data
from glob import glob
import os
import os.path
import SimpleITK as sitk
import pandas as pd
from params import *
import matplotlib.pyplot as plt

class XrayCTPADataset(data.Dataset):
    def __init__(self, root='.', target=None, mode="train", augmentation=False):
        if target is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))


        self.mode = mode
        self.root = root
        self.cts = self.root + 'preprocessed_data/XRayCTPA/CTPA_256/'

        self.xrays = self.root + 'preprocessed_data/XRayCTPA/XRay_preprocessed/'
        if target is not None:
            self.data = pd.read_csv(root + target)
            cts_list = [ct.replace(".npy","") for ct in os.listdir(self.cts)]
            xrays_list = [xr.replace(".npy","") for xr in os.listdir(self.xrays)]
            self.data[CT_ACCESSION_COL] = self.data[CT_ACCESSION_COL].astype(int)
            self.data = self.data[self.data[CT_ACCESSION_COL].isin(cts_list)]
            self.data[XRAY_ACCESSION_COL] = self.data[XRAY_ACCESSION_COL].astype(int)
            self.data = self.data[self.data[XRAY_ACCESSION_COL].isin(xrays_list)]
            self.data = self.data.reset_index()
        self.augmentation = augmentation
        self.VAE = True
        self.text_label = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ct_accession = self.data.loc[idx, CT_ACCESSION_COL]
        cxr_accession = self.data.loc[idx, XRAY_ACCESSION_COL]
        label = self.data.loc[idx, LABEL_COL]
        # label = F.one_hot(label.view(-1).to(torch.int64), num_classes=2).to(torch.float32)
        if not self.text_label:
            label = torch.tensor(label).type(torch.DoubleTensor)
            label = label.reshape(1)
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

        if self.mode == "train" or self.mode == "test":
            return {'ct': ctout, 'cxr': xray, 'target': label} 
        else: #if self.mode == "infer"
            return {'ct': ctout, 'cxr': xray, 'ct_accession': ct_accession, 'cxr_accession': cxr_accession}
