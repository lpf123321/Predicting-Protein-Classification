import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ProteinRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        """embedding层+双向LSTM+全连接层"""
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed)  # hidden: (num_directions, batch, hidden_dim)
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        hidden_fw = hidden[-2]  # 最后一层正向
        hidden_bw = hidden[-1]  # 最后一层反向
        hidden = torch.cat((hidden_fw, hidden_bw), dim=1)
        # hidden = hidden[-1]
        out = self.fc(hidden)
        return out
