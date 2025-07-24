import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """CNN backbone for text classification"""

    def __init__(self, config):
        super(CNNBackbone, self).__init__()

        self.vocab_size = config.get('vocab_size', 1000)
        self.embedding_dim = config.get('embedding_dim', 64)
        self.num_filters = config.get('num_filters', [100, 100, 100])
        self.kernel_sizes = config.get('kernel_sizes', [3, 4, 5])
        self.dropout_rate = config.get('dropout', 0.3)

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(self.embedding_dim, num_filter, kernel_size)
            for kernel_size, num_filter in zip(self.kernel_sizes, self.num_filters)
        ])

        # Output dimension
        self.output_dim = sum(self.num_filters)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        Args:
            x: Input sequences [batch_size, seq_len]
        Returns:
            output: Pooled representation [batch_size, output_dim]
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution + ReLU
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, conv_seq_len]

            # Global max pooling
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            pooled = pooled.squeeze(2)  # [batch_size, num_filters]

            conv_outputs.append(pooled)

        # Concatenate all conv outputs
        output = torch.cat(conv_outputs, dim=1)  # [batch_size, sum(num_filters)]
        output = self.dropout(output)

        return output

    def get_output_dim(self):
        """Get output dimension for classification head"""
        return self.output_dim
