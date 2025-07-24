import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for language detection"""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        self.input_dim:int = config.get('backbone_output_dim', 128)
        self.num_classes:int = config.get('num_classes', 235)  # Default WiLi classes
        self.dropout_rate:float = config.get('dropout', 0.2)
        self.device:str = config.get('device', 'cpu')

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def set_output_dim(self, num_classes:int):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.input_dim, self.num_classes).to(self.device)

    def forward(self, x):
        """
        Args:
            x: Output from backbone [batch_size, backbone_output_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
