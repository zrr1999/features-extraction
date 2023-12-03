import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from rich.progress import Progress, track
from tools.dataset import  FaceFeaturesDataset, split_dataset, split_dataset_by_class
from tools.utils import load_features, calculate_accuracy, get_all_methods
from rich import print
from models.simple import FaceRecognitionModel



def train_and_eval(model: nn.Module, num_epochs: int, detection_method:str, recognition_method:str, train_dataset, test_dataset):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns())  as progress:
        best_loss = float("inf")
        epochs_without_improvement = 0
        task = progress.add_task(f"[green]Using {detection_method} and {recognition_method}...", total=num_epochs, loss_value=best_loss)
        for epoch in range(num_epochs):
            loss_value = float("inf")
            loss_value_list = []
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_value_list.append(loss.item())
            loss_value = sum(loss_value_list)/len(loss_value_list)
            progress.update(task, advance=1, loss_value=loss_value)
            if best_loss>loss_value:
                best_loss=loss_value
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 5:
                    print("Early stopping!")
                    break

    train_accuracy = calculate_accuracy(model, train_data_loader)
    test_accuracy = calculate_accuracy(model, test_data_loader)

    return train_accuracy, test_accuracy




num_epochs = 1000
batch_size = 4096
num_classes = 130
dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"
use_cuda = True

criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
for detection_method, recognition_method, feature_size in get_all_methods(ignore_methods="dlib"):
    features_dict = load_features(features_path, detection_method, recognition_method)
    mean_accuracy = 0
    for i, (train_dataset, test_dataset) in enumerate(split_dataset_by_class(FaceFeaturesDataset(features_dict, use_cuda=use_cuda), folds=10)):
        model = FaceRecognitionModel(feature_size, num_classes)
        if use_cuda:
            model = model.cuda()
        train_accuracy, test_accuracy = train_and_eval(model, num_epochs, detection_method, recognition_method, train_dataset, test_dataset)
        mean_accuracy += test_accuracy
        print(f'Accuracy in {detection_method}-{recognition_method}({i+1}/10): test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%')
    mean_accuracy/=10
    print(f'Mean Accuracy in {detection_method}-{recognition_method}: {mean_accuracy:.2f}%')
    mean_accuracies[(detection_method, recognition_method)] = mean_accuracy

print(mean_accuracies)
