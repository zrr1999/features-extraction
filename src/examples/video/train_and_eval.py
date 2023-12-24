from __future__ import annotations

import os

import torch
from loguru import logger
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader

from extractor.dataset.utils import split_dataset_by_class
from extractor.dataset.video import VideoFeaturesDataset
from extractor.utils import EarlyStopper, calculate_accuracy, calculate_f1_score, load_features
from extractor.vision.utils import get_all_video_methods
from model.simple import SimpleConv1dModel

logger.add("logs/train_and_eval.log", rotation="10 MB")


def train_and_eval(
    model: nn.Module,
    num_epochs: int,
    method: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
        stopper = EarlyStopper(10)

        task = progress.add_task(
            f"[green]Using {method} pkl to train",
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
batch_size = 512
num_classes = 7
use_cuda = True
features_path = "./datasets/features/video"

criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
mean_f1_scores = {}
for method, feature_size in get_all_video_methods():
    features_dict = load_features(features_path, f"{method}_features")
    mean_accuracy = 0
    mean_f1_score = 0
    for i, (train_dataset, test_dataset) in enumerate(
        split_dataset_by_class(VideoFeaturesDataset(features_dict, use_cuda=use_cuda), folds=10)
    ):
        if os.path.exists(f"checkpoints/{method}_{i}.pth"):
            logger.info(f"skip: checkpoints/{method}_{i}.pth exists")
            continue

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleConv1dModel(10, feature_size, num_classes)
        if use_cuda:
            model = model.cuda()
        train_accuracy, test_accuracy = train_and_eval(
            model,
            num_epochs,
            method,
            train_data_loader,
            test_data_loader,
        )

        checkpoint_dir = f"checkpoints/models/{method}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            model,
            f"{checkpoint_dir}/{i}.pth",
        )
        logger.info(f"Model saved: {checkpoint_dir}/{i}.pth")

        mean_accuracy += test_accuracy
        logger.info(f"Accuracy in {method}({i+1}/10): test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%")
        f1_score = calculate_f1_score(model, test_data_loader)
        mean_f1_score += f1_score
        logger.info(f"F1 Score: {f1_score:.2f}")
    mean_accuracy /= 10
    mean_f1_score /= 10
    logger.info(f"Mean Accuracy in {method}: {mean_accuracy:.2f}%")
    mean_accuracies[method] = mean_accuracy
    mean_f1_scores[method] = mean_f1_score


logger.info(mean_accuracies)
logger.info(mean_f1_scores)
