import os
import glob

import torch

import pandas as pd

from torch.utils.data import Dataset

class Motion3DDataset(Dataset):
    def __init__(self, dataset_root):
        self.files = files = glob.glob(os.path.join(dataset_root, '*.pkl'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = pd.read_pickle(self.files[idx])
        
        data = torch.from_numpy(x['data'])
        mask = torch.from_numpy(x['mask']).int()

        return data, mask