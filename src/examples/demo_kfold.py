from __future__ import annotations

import torch
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader

from extractor.dataset.face import FaceFeaturesDataset
from extractor.dataset.utils import split_dataset_by_class
from extractor.utils import calculate_accuracy, load_features

dataset_path = "./datasets/Face-Dataset/UCEC-Face"
features_path = "./datasets/features/vision"
detection_methods = ["dlib", "mediapipe","mtcnn","ssd"]
recognition_methods = ["Facenet", "ArcFace", "OpenFace", "DeepFace"]


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


criterion = nn.CrossEntropyLoss()

feature_size = 128
num_classes = 130
detection_method = "dlib"
recognition_method = "Facenet"
features_dict = load_features(features_path, detection_method, recognition_method)


for i, (train_dataset, test_dataset) in enumerate(split_dataset_by_class(FaceFeaturesDataset(features_dict), folds=10)):
    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = FaceRecognitionModel(feature_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
        task = progress.add_task(
            f"[green]Using {detection_method} and {recognition_method}...",
            total=num_epochs,
            loss_value=-1,
        )
        for epoch in range(num_epochs):
            loss_values = []
            for features, labels in train_data_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
            progress.update(task, advance=1, loss_value=sum(loss_values) / len(loss_values))

            if epoch % 50 == 0:
                train_accuracy = calculate_accuracy(model, train_data_loader)
                test_accuracy = calculate_accuracy(model, test_data_loader)
                print(
                    f"Test Accuracy in {detection_method}-{recognition_method}({i+1}/10): test_accuracy: {test_accuracy:.2f}%, train_accuracy: {train_accuracy:.2f}%"
                )
