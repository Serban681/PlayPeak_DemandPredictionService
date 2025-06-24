import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, extra_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.extra_fc = nn.Linear(extra_size, 8)
        self.combined_fc = nn.Linear(hidden_size + 8, output_size)

    def forward(self, x_seq, x_extra):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]  # last time step
        extra_feat = torch.relu(self.extra_fc(x_extra))
        combined = torch.cat((lstm_feat, extra_feat), dim=1)
        return self.combined_fc(combined)
