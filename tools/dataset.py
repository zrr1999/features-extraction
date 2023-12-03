import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from rich.progress import Progress, track
import random
from torch.utils.data import Subset, ConcatDataset

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
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)-1

    def __len__(self):
        return len(self.items)
    

def split_dataset(dataset, *, folds=10):
    dataset_size = len(dataset)
    single_split_size = dataset_size // folds
    for i in range(folds):
        # 生成当前折的测试集和训练集的索引
        test_indices = list(range(i * single_split_size, (i + 1) * single_split_size))
        train_indices = list(range(0, i * single_split_size)) + list(range((i + 1) * single_split_size, dataset_size))
        
        # 根据索引创建数据子集
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        yield train_subset, test_subset


def split_dataset_by_class(dataset, num_classes=130, *, folds=10):
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for i in range(folds):
        class_train_subsets = []
        class_test_subsets = []
        for indices in class_indices:
            class_subset = Subset(dataset, indices)
            subset_size = len(class_subset)
            single_split_size = subset_size // folds
            
            test_indices = list(range(i * single_split_size, (i + 1) * single_split_size))
            train_indices=list(range(0, i * single_split_size)) + list(range((i + 1) * single_split_size, subset_size))

            class_train_subsets.append(Subset(class_subset, train_indices))
            class_test_subsets.append(Subset(class_subset, test_indices))

        # 根据索引创建数据子集
        train_subset = ConcatDataset(class_train_subsets)
        test_subset = ConcatDataset(class_test_subsets)
        yield train_subset, test_subset
    
