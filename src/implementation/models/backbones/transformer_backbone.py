import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, embedding_dim, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerBackbone(nn.Module):
    """Lightweight Transformer backbone for text classification"""

    def __init__(self, config):
        super(TransformerBackbone, self).__init__()

        self.vocab_size = config.get('vocab_size', 1000)
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 3)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.dropout_rate = config.get('dropout', 0.1)
        self.max_length = config.get('max_length', 200)

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(self.embedding_dim, self.max_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output dimension
        self.output_dim = self.embedding_dim

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Create padding mask
        padding_mask = (x == 0)  # Assuming 0 is padding token

        # Embedding + positional encoding
        embedded = self.embedding(x) * math.sqrt(self.embedding_dim)  # [batch_size, seq_len, embedding_dim]
        embedded = self.pos_encoding(embedded)
        embedded = self.dropout(embedded)

        # Transformer encoding
        encoded = self.transformer(embedded, src_key_padding_mask=padding_mask)  # [batch_size, seq_len, embedding_dim]

        # Global average pooling (excluding padding tokens)
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        masked_encoded = encoded * mask

        # Sum and normalize by actual sequence length
        output = masked_encoded.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch_size, embedding_dim]

        return output

    def get_output_dim(self):
        """Get output dimension for classification head"""
        return self.output_dim