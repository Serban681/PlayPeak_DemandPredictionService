import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, extra_size, output_size, nhead=2, num_layers=2):
        super(TransformerModel, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.extra_fc = nn.Linear(extra_size, 8)

        self.combined_fc = nn.Linear(input_size + 8, output_size)

    def forward(self, x_seq, x_extra):
        x_seq = x_seq.transpose(0, 1)
        transformer_out = self.transformer_encoder(x_seq)

        transformer_feat = transformer_out[-1, :, :]

        extra_feat = torch.relu(self.extra_fc(x_extra))

        combined = torch.cat((transformer_feat, extra_feat), dim=1)

        return self.combined_fc(combined)
