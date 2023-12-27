from __future__ import annotations

import torch
from torch import nn


class BaseModel(nn.Module):
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)


class SimpleModel(BaseModel):
    def __init__(self, feature_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


class Conv1dModel(BaseModel):
    def __init__(self, input_size, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((feature_size - 2) // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.input_size = input_size

    def forward(self, x):
        x = x[:, : self.input_size]
        x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], x.shape[2], device=x.device)], dim=1)

        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * ((self.feature_size - 2) // 2))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Conv1dSoftmaxModel(BaseModel):
    def __init__(self, input_size, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.final = nn.Sequential(
            nn.Linear(64 * ((feature_size - 2) // 2), 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=-1),
        )
        self.input_size = input_size

    def forward(self, x):
        x = x[:, : self.input_size]
        x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], x.shape[2], device=x.device)], dim=1)

        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * ((self.feature_size - 2) // 2))
        x = self.final(x)
        return x


class Conv1dLogSoftmaxModel(BaseModel):
    def __init__(self, input_size, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.final = nn.Sequential(
            nn.Linear(64 * ((feature_size - 2) // 2), 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=-1),
        )
        self.input_size = input_size

    def forward(self, x):
        x = x[:, : self.input_size]
        x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], x.shape[2], device=x.device)], dim=1)

        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * ((self.feature_size - 2) // 2))
        x = self.final(x)
        return x


class Conv1dSigmoidModel(BaseModel):
    def __init__(self, input_size, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.final = nn.Sequential(
            nn.Linear(64 * ((feature_size - 2) // 2), 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
        self.input_size = input_size

    def forward(self, x):
        x = x[:, : self.input_size]
        x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], x.shape[2], device=x.device)], dim=1)

        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * ((self.feature_size - 2) // 2))
        x = self.final(x)
        return x


class LSTMModel(BaseModel):
    def __init__(self, feature_size, hidden_dim, num_classes, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden_dim, 5, batch_first=True, bidirectional=bidirectional)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.final(x)
        return x


class GRUModel(BaseModel):
    def __init__(self, feature_size, hidden_dim, num_classes, bidirectional=True):
        super().__init__()
        self.lstm = nn.GRU(feature_size, hidden_dim, 5, batch_first=True, bidirectional=bidirectional)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.final(x)
        return x


# class Conv1dLSTM(BaseModel):
#     def __init__(
#         self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
#     ):
#         super().__init__()
#         self.encoder = Encoder(latent_dim)
#         self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
#         self.output_layers = nn.Sequential(
#             nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim, momentum=0.01),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes),
#             nn.Softmax(dim=-1),
#         )
#         self.attention = attention
#         self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         x = self.encoder(x)
#         x = x.view(batch_size, seq_length, -1)
#         x = self.lstm(x)
#         if self.attention:
#             attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
#             x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
#         else:
#             x = x[:, -1]
#         return self.output_layers(x)
