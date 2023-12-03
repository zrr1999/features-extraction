from torch import nn


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
