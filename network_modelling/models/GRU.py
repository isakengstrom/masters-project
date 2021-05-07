import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    First and simplest GRU net
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, dropout=.5):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)

        # out: batch_size, seq_length, hidden_size
        out, _ = self.gru(x, (h0, c0))

        # Decode the hidden state of the last time step (many to one)
        out = out[:, -1, :]

        # out: (n,
        out = self.fc(out)

        return out
