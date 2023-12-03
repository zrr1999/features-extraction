import torch
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, track
from tools.dataset import FaceFeaturesDataset
from tools.utils import load_features


dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"
detection_methods = ["dlib", "mediapipe"]
recognition_methods = ["Facenet", "ArcFace", "OpenFace", "DeepFace"]

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
recognition_method = "Facenet" 
features_dict = load_features(features_path, detection_method, recognition_method)
dataset = FaceFeaturesDataset(features_dict)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FaceRecognitionModel(feature_size, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
with Progress("[red](Loss: {task.fields[loss_value]:.8f})", *Progress.get_default_columns())  as progress:
    task = progress.add_task(f"[green]Using {detection_method} and {recognition_method}...", total=num_epochs, loss_value=-1)
    for epoch in range(num_epochs):
        loss_values = []
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
        progress.update(task, advance=1, loss_value=sum(loss_values)/len(loss_values))


def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    accuracy = 100 * correct / total
    return accuracy

test_accuracy = calculate_accuracy(model, data_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
