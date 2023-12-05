from __future__ import annotations

import os

import torch
from loguru import logger
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader

from models.simple import FaceRecognitionModel
from tools.dataset import FaceFeaturesDataset, split_dataset_by_class
from tools.utils import calculate_accuracy, calculate_f1_score, get_all_methods, load_features

logger.add("logs/train_and_eval.log", rotation="10 MB")


class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_scores = {}

    def update(self, **kwargs: float):
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = (float("-inf"), 0)
            if value > self.best_scores[key][0]:
                self.best_scores[key] = (value, self.best_scores[key][1])
            else:
                self.best_scores[key] = (self.best_scores[key][0], self.best_scores[key][1] + 1)
                if self.best_scores[key][1] >= self.patience:
                    logger.info(f"Early stopping by {key}!")
                    return True
        return False


def train_and_eval(
    model: nn.Module,
    num_epochs: int,
    detection_method: str,
    recognition_method: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
        stopper = EarlyStopper(5)

        task = progress.add_task(
            f"[green]Using {detection_method} and {recognition_method} pkl to train",
            total=num_epochs,
            loss_value=float("inf"),
        )
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
            loss_value = sum(loss_value_list) / len(loss_value_list)
            progress.update(task, advance=1, loss_value=loss_value)
            if epoch % 10 == 0:
                test_accuracy = calculate_accuracy(model, test_data_loader)
                if stopper.update(loss=-loss_value, accuracy=test_accuracy):
                    break

    train_accuracy = calculate_accuracy(model, train_data_loader)
    test_accuracy = calculate_accuracy(model, test_data_loader)

    return train_accuracy, test_accuracy


num_epochs = 10000
batch_size = 4096 * 4
num_classes = 130
dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"
use_cuda = True

criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
mean_f1_scores = {}
for detection_method, recognition_method, feature_size in get_all_methods():
    features_dict = load_features(features_path, detection_method, recognition_method)
    mean_accuracy = 0
    mean_f1_score = 0
    for i, (train_dataset, test_dataset) in enumerate(
        split_dataset_by_class(FaceFeaturesDataset(features_dict, use_cuda=use_cuda), folds=10)
    ):
        if os.path.exists(f"checkpoints/{detection_method}_{recognition_method}_{i}.pth"):
            logger.info(f"skip: checkpoints/{detection_method}_{recognition_method}_{i}.pth exists")
            continue

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = FaceRecognitionModel(feature_size, num_classes)
        if use_cuda:
            model = model.cuda()
        train_accuracy, test_accuracy = train_and_eval(
            model,
            num_epochs,
            detection_method,
            recognition_method,
            train_data_loader,
            test_data_loader,
        )

        checkpoint_dir = f"checkpoints/models/{detection_method}_{recognition_method}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            model,
            f"{checkpoint_dir}/{i}.pth",
        )
        logger.info(f"Model saved: {checkpoint_dir}/{i}.pth")

        mean_accuracy += test_accuracy
        logger.info(
            f"Accuracy in {detection_method}-{recognition_method}({i+1}/10): test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%"
        )
        f1_score = calculate_f1_score(model, test_data_loader)
        mean_f1_score += f1_score
        logger.info(f"F1 Score: {f1_score:.2f}")
    mean_accuracy /= 10
    mean_f1_score /= 10
    logger.info(f"Mean Accuracy in {detection_method}-{recognition_method}: {mean_accuracy:.2f}%")
    mean_accuracies[(detection_method, recognition_method)] = mean_accuracy
    mean_f1_scores[(detection_method, recognition_method)] = mean_f1_score


logger.info(mean_accuracies)
logger.info(mean_f1_scores)
