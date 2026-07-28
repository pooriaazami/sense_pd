import os
import glob

from torch.utils.data import Dataset

import pandas as pd

class CarePDDataset(Dataset):
    def __init__(self, root, datasets=None):
        files = glob.glob(os.path.join(root, '*.pkl'))

        if datasets:
            files = list(filter(lambda x: x.replace(root, '').replace('/', '').split('_')[0] in datasets, files))

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, x):
        path = self.files[0]

        file = pd.read_pickle(path)
        return file