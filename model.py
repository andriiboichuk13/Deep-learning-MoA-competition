import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, channels: int = 256, length: int = 16):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels
        self.length = length

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(input_dim, channels * length)
        self.dropout2 = nn.Dropout(0.2)

        self.norm2 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1)
        self.pool1 = nn.MaxPool1d(kernel_size = 2)

        self.norm3 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.output = nn.Linear(channels * (length // 4), output_dim)
        # self.init_weights()

    def forward(self, x):
        x = self.dropout2(F.relu(self.linear1(self.dropout1(self.norm1(x)))))
        x = x.view(-1, self.channels, self.length)
        skip = x
        x = F.relu(self.conv1(self.norm2(x)))
        x = x + skip
        x = self.pool1(x)
        skip = x
        x = F.relu(self.conv2(self.norm3(x)))
        x = x + skip
        x = self.pool2(x).flatten(1)
        output = self.output(x)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)



