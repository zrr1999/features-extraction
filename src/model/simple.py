from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class Conv1dLogSoftmaxAttentionModel(BaseModel):
    def __init__(self, input_size, feature_size, num_classes, num_heads=16):
        super(Conv1dLogSoftmaxAttentionModel, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 使用 nn.MultiheadAttention 实现多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)

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

        x = x.permute(2, 0, 1)  # 将维度调整为 (seq_len, batch_size, embed_dim)
        attention_output, _ = self.multihead_attention(x, x, x)
        x = attention_output.permute(1, 2, 0)  # 调整维度为 (batch_size, embed_dim, seq_len)

        x = x.reshape(-1, 64 * ((self.feature_size - 2) // 2))
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


class LSTMModelWithMultiheadAttention(BaseModel):
    def __init__(self, feature_size, hidden_dim, num_classes, bidirectional=True, num_heads=8):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden_dim, 5, batch_first=True, bidirectional=bidirectional)
        self.multihead_attention = nn.MultiheadAttention(hidden_dim * 2 if bidirectional else hidden_dim, num_heads)
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output.permute(1, 0, 2)  # Change the dimensions for MultiheadAttention
        attended_output, _ = self.multihead_attention(lstm_output, lstm_output, lstm_output)
        attended_output = attended_output.permute(1, 0, 2)  # Change the dimensions back
        attended_output = attended_output.mean(dim=1)
        output = self.final(attended_output)
        return output


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
class LSTMModel_vl(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel_vl, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        # 对序列按长度进行排序
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sorted_idx]
        # 使用 pack_padded_sequence 对序列进行压缩
        packed_sequence = pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True)
        # 通过 LSTM 处理压缩后的序列 2
        packed_output, _ = self.lstm(packed_sequence)
        # 使用 pad_packed_sequence 对序列进行解压缩
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #print(output.shape,"---------output.shape")
        # 恢复原始排序
        _, original_idx = torch.sort(sorted_idx)
        output = output[original_idx]
        # 获取最后一个时间步的输出
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        #print(last_output,last_output.shape,"---------last_output.shape")
        # 使用全连接层进行分类
        x = self.fc1(last_output)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        #print(x,x.shape)
        return x

class GRUModel_vl(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel_vl, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        #self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self, x, lengths):
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sorted_idx]
        # 使用 pack_padded_sequence 对序列进行压缩
        packed_sequence = pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True)
        # 通过 GRU 处理压缩后的序列
        packed_output, _ = self.gru(packed_sequence)
        # 使用 pad_packed_sequence 对序列进行解压缩
      
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 恢复原始排序
        _, original_idx = torch.sort(sorted_idx)
        output = output[original_idx]
        # 获取最后一个时间步的输出
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        x = self.fc1(last_output)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
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
