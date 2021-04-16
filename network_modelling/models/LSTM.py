import os
import torch
import torch.nn as nn
#import torch.nn.functional as F


# https://towardsdatascience.com/metric-learning-loss-functions-5b67b3da99a5

class LSTM(nn.Module):
    """
    First and simplest LSTM net
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Should it be this ??
        #self.fc = nn.Linear(hidden_size*seq_len, num_classes)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float).to(self.device)

        # out: batch_size, seq_length, hidden_size
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step (many to one)
        out = out[:, -1, :]
        #out = out.reshape(out.shape[0], -1)

        # out: (n,
        out = self.fc(out)

        return out


class LSTM_2(nn.Module):
    """
    Second LSTM net, with a few more layers
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM_2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float).to(self.device)

        # out: batch_size, seq_length, hidden_size
        output, (hn, cn) = self.lstm(x, (h0, c0))

        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

