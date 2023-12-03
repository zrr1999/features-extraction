import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from rich.progress import Progress, track

def load_features(features_path, detection_method, recognition_method):
    file_path = f"{features_path}/{detection_method}_{recognition_method}.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


class FaceFeaturesDataset(Dataset):
    def __init__(self, features_dict):
        self.features_dict = features_dict
        self.items = list(features_dict.items())

    def __getitem__(self, index):
        (label, _), (_, features) = self.items[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.items)