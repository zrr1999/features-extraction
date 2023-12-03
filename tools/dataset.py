import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from rich.progress import Progress, track


class FaceFeaturesDataset(Dataset):
    def __init__(self, features_dict):
        self.features_dict = features_dict
        self.items = list(features_dict.items())

    def __getitem__(self, index):
        (label, _), (_, features) = self.items[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.items)
    

def split_dataset(dataset, num_classes=130, *, folds=10):

    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
        
    dataset_size = len(dataset)
    single_split_size = dataset_size // folds
    splits = [single_split_size] * 9
    splits.append(dataset_size - sum(splits))
    for i in range(folds):
        # 生成当前折的测试集和训练集的索引
        test_indices = list(range(i * single_split_size, (i + 1) * single_split_size))
        train_indices = list(range(0, i * single_split_size)) + list(range((i + 1) * single_split_size, dataset_size))
        
        # 根据索引创建数据子集
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        
        yield train_subset, test_subset
