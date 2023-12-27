from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FaceFeaturesDataset(Dataset):
    def __init__(self, features_dict, *, use_cuda=False):
        self.features_dict = features_dict
        self.use_cuda = use_cuda
        self.items = list(features_dict.items())

    def __getitem__(self, index):
        (label, _), (_, features) = self.items[index]
        if self.use_cuda:
            return torch.tensor(features, dtype=torch.float32).cuda(), torch.tensor(label, dtype=torch.int64).cuda() - 1
        else:
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.int64) - 1

    def __len__(self):
        return len(self.items)
