from __future__ import annotations

import torch
from rich.progress import Progress
from torch.utils.data import DataLoader

from dataset.face import FaceFeaturesDataset
from utils import calculate_accuracy, calculate_f1_score, load_features

dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"


class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, feature_size, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.fc = torch.nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


feature_size = 128
num_classes = 130
detection_method = "dlib"
recognition_method = "OpenFace"
features_dict = load_features(features_path, detection_method, recognition_method)
dataset = FaceFeaturesDataset(features_dict)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FaceRecognitionModel(feature_size, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns()) as progress:
    task = progress.add_task(
        f"[green]Using {detection_method} and {recognition_method}...",
        total=num_epochs,
        loss_value=-1,
    )
    for _ in range(num_epochs):
        loss_values = []
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
        progress.update(task, advance=1, loss_value=sum(loss_values) / len(loss_values))

        test_accuracy = calculate_accuracy(model, data_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        f1_score = calculate_f1_score(model, data_loader)
        print(f"F1 Score: {f1_score*100:.2f}")
