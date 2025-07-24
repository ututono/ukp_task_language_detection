import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):

    def __init__(self, config):
        super(LSTMBackbone, self).__init__()

        self.vocab_size = config.get('vocab_size', 1000)
        self.embedding_dim = config.get('embedding_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.bidirectional = config.get('bidirectional', True)
        self.dropout_rate = config.get('dropout', 0.2)

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            batch_first=True
        )

        # Output dimension calculation
        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim * directions]

        # Use last hidden state (both directions if bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            output = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim * 2]
        else:
            output = hidden[-1]  # [batch_size, hidden_dim]

        return output

    def get_output_dim(self):
        """Get output dimension for classification head"""
        return self.output_dim
