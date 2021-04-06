import os
import torch
import torch.nn as nn
#import torch.nn.functional as F


# https://towardsdatascience.com/metric-learning-loss-functions-5b67b3da99a5

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, use_cuda):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step (many to one)
        out = out[:, -1, :]

        # out: (n,
        out = self.fc(out)

        return out
