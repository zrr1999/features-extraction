from __future__ import annotations

import os

import opensmile
import torch
from loguru import logger
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader

from extractor.audio.utils import get_all_audio_methods
from extractor.dataset.audio import AudioFeaturesDataset
from extractor.utils import EarlyStopper, calculate_accuracy, calculate_f1_score, load_features
from model.simple import SimpleAudioModel

logger.add("checkpoints/logs/audio/train_and_eval.log", rotation="10 MB")


def train_and_eval(
    model: nn.Module,
    num_epochs: int,
    dataset_name: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
        stopper = EarlyStopper(5)

        task = progress.add_task(
            f"[green]Using {dataset_name} pkl to train",
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


num_epochs = 1000
batch_size = 4096
num_classes = 5
features_path = "./datasets/features/audio"
use_cuda = True

criterion = nn.CrossEntropyLoss()
if use_cuda:
    criterion = criterion.cuda()
mean_accuracies = {}
mean_f1_scores = {}
for smile, feature_size in get_all_audio_methods():
    if smile.feature_level == opensmile.FeatureLevel.Functionals:
        continue
    train_features_dict = load_features(features_path, f"train_{smile.feature_set}_{smile.feature_level}")
    test_features_dict = load_features(features_path, f"test_{smile.feature_set}_{smile.feature_level}")
    mean_accuracy = 0
    mean_f1_score = 0

    checkpoint_dir = f"checkpoints/models/audio/"
    checkpoint_path = f"{checkpoint_dir}/{smile.feature_set}_{smile.feature_level}.pth"

    train_dataset = AudioFeaturesDataset(train_features_dict, use_cuda=use_cuda)
    test_dataset = AudioFeaturesDataset(test_features_dict, use_cuda=use_cuda)

    if os.path.exists(checkpoint_path):
        logger.info(f"skip: {checkpoint_path} exists")
        continue

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleAudioModel(256, feature_size, num_classes)
    if use_cuda:
        model = model.cuda()
    train_accuracy, test_accuracy = train_and_eval(
        model,
        num_epochs,
        f"{smile.feature_set}_{smile.feature_level}",
        train_data_loader,
        test_data_loader,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        model,
        checkpoint_path,
    )
    logger.info(f"Model saved: {checkpoint_path}")

    mean_accuracy += test_accuracy
    logger.info(
        f"Accuracy in {smile.feature_set}-{smile.feature_level}: test: {test_accuracy:.2f}%, train: {train_accuracy:.2f}%"
    )
    f1_score = calculate_f1_score(model, test_data_loader)
    mean_f1_score += f1_score
    logger.info(f"F1 Score: {f1_score:.2f}")

    logger.info(f"Mean Accuracy in {smile.feature_set}-{smile.feature_level}: {mean_accuracy:.2f}%")
    mean_accuracies[(smile.feature_set, smile.feature_level)] = mean_accuracy
    mean_f1_scores[(smile.feature_set, smile.feature_level)] = mean_f1_score


logger.info(mean_accuracies)
logger.info(mean_f1_scores)
