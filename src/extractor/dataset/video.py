from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


def emotion2int(emotion: str):
    emotion2int = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
    return emotion2int[emotion]


class VideoFeaturesDataset(Dataset):
    def __init__(self, features_dict, *, use_cuda=False):
        self.features_dict = features_dict
        self.use_cuda = use_cuda
        self.items = list(features_dict.items())

    def __getitem__(self, index):
        (label, _), features = self.items[index]
        features = torch.tensor(features[:10], dtype=torch.float32)
        features = torch.cat([features, torch.zeros(10 - features.shape[0], features.shape[1])])
        if self.use_cuda:
            return features.cuda(), torch.tensor(label, dtype=torch.int64).cuda()
        else:
            return features, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.items)
