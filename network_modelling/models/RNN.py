import torch
import torch.nn as nn


class GenNet(nn.Module):
    """
    A General Net for RNNs. The available Nets are RNN, GRU and LSTM

    - Add bidirectionality through the bidirectional param
    - Add dropout with the dropout param,

    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, dropout: float = .5,
                 bidirectional=False, net_type="lstm"):

        super(GenNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.use_cell_state = False

        net_type = net_type.lower()
        if net_type == "rnn":
            self.net = nn.RNN
        elif net_type == "gru":
            self.net = nn.GRU
        elif net_type == "lstm":
            self.use_cell_state = True
            self.net = nn.LSTM
        else:
            raise Exception("Invalid Recurrent neural network type")

        self.net = self.net(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        if bidirectional:
            self.num_layers = self.num_layers * 2
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)

        # Use cell state for gated cells, such as GRU and LSTM, else use only the hidden state
        if self.use_cell_state:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)

            # out: batch_size, seq_length, hidden_size
            out, _ = self.net(x, (h0, c0))

        else:
            # out: batch_size, seq_length, hidden_size
            out, _ = self.net(x, h0)

        # Decode the hidden state of the last time step (many to one)
        out = out[:, -1, :]

        # out: (n,
        out = self.fc(out)

        return out
