import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from rich.progress import Progress, track
from tools.dataset import  FaceFeaturesDataset, split_dataset, split_dataset_by_class
from tools.utils import load_features, calculate_accuracy, get_all_methods
from loguru import logger

dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"
detection_methods = ["dlib", "mediapipe"]
recognition_methods = ["Facenet", "ArcFace", "OpenFace", "DeepFace", "Facenet512", "VGG-Face"]
use_cuda = True

class FaceRecognitionModel(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


def train_and_eval(detection_method, recognition_method, feature_size, train_dataset, test_dataset):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = FaceRecognitionModel(feature_size, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000
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
                if epochs_without_improvement >= 10:
                    logger.info("Early stopping!")
                    break

    train_accuracy = calculate_accuracy(model, train_data_loader)
    test_accuracy = calculate_accuracy(model, test_data_loader)

    return train_accuracy, test_accuracy




criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
for detection_method, recognition_method, feature_size in get_all_methods():
    batch_size = 4096
    num_classes = 130
    features_dict = load_features(features_path, detection_method, recognition_method)

    mean_accuracy = 0
    for i, (train_dataset, test_dataset) in enumerate(split_dataset_by_class(FaceFeaturesDataset(features_dict, use_cuda=use_cuda), folds=10)):
        train_accuracy, test_accuracy = train_and_eval(detection_method, recognition_method, feature_size, train_dataset, test_dataset)
        mean_accuracy += test_accuracy
        logger.info(f'Accuracy in {detection_method}-{recognition_method}({i+1}/10): test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%')
    mean_accuracy/=10
    logger.info(f'Mean Accuracy in {detection_method}-{recognition_method}: {mean_accuracy:.2f}%')
    mean_accuracies[(detection_method, recognition_method)] = mean_accuracy

logger.info(mean_accuracies)
