import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
        A regular LSTM with multi-layers and dropout layers.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, horizon=1, batch_first=True):
        super(LSTM, self).__init__()
        self.horizon = horizon
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size * horizon, output_size * horizon)

    def forward(self, x):
        # x(N, dim, seq_length)
        out, _ = self.rnn(x.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        N = out.size()[0]
        return self.linear(out[:, :, -self.horizon:].reshape(N, -1))


class GRU(nn.Module):
    """
        A regular GRU with multi-layers and dropout layers.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2, horizon=1, batch_first=True):
        super(GRU, self).__init__()
        self.horizon = horizon
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size * horizon, output_size * horizon)

    def forward(self, x):
        out, _ = self.rnn(x.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        N = out.size()[0]
        return self.linear(out[:, :, -self.horizon:].reshape(N, -1))
