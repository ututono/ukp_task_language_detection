import logging
from typing import List

import torch
import torch.nn as nn

from src.core.abstractions.model import AbstractModel
from src.core.entities.config import DLModelConfig

from src.infrastructure.utils.constants import DetectorModelType

logger = logging.getLogger(__name__)


class BaseDeepLearningLanguageDetector(nn.Module, AbstractModel):
    """
    Base calss for deep learning language detectors.
    """

    @classmethod
    def build_config(cls, config):
        model_config = config.get('model_config', {})
        cfg = config.get('cfg', None)
        return DLModelConfig(
            backbone_config=model_config,
            num_classes=config.get('num_classes', 235),
            model_type=model_config.get('model_type', None),
            tokeinizer_config=config.get('tokenizer_config', None),
            device=cfg.local.get('device', 'cpu') if cfg else 'cpu',
            cfg=cfg,
        )

    def build(self):
        pass

    def __init__(self, config: DLModelConfig):
        nn.Module.__init__(self)

        self._model_type = config.get('model_type')
        self._num_classes = config.get('num_classes')

        self.text_encoder = self._build_text_encoder(config)
        self.backbone = self._build_backbone(config.get('backbone_config'))

        self._backbone_output_dim: int = self.backbone.get_output_dim()
        config.backbone_output_dim = self._backbone_output_dim
        self.classification_head = self._build_classification_head(config)

        AbstractModel.__init__(self, config)

    def _build_text_encoder(self, config):
        from src.implementation.models.components.text_encoder import TextEncoder
        text_encoder_config = TextEncoder.build_config(config.tokeinizer_config)
        return TextEncoder(text_encoder_config)

    def _build_backbone(self, config):
        model_type = self._model_type

        if model_type == DetectorModelType.CNN.value.lower():
            from src.implementation.models.backbones.cnn_backbone import CNNBackbone
            return CNNBackbone(config)

        elif model_type == DetectorModelType.LSTM.value.lower():
            from src.implementation.models.backbones.lstm_backbone import LSTMBackbone
            return LSTMBackbone(config)

        elif model_type == DetectorModelType.TRANSFORMER.value.lower():
            from src.implementation.models.backbones.transformer_backbone import TransformerBackbone
            return TransformerBackbone(config)

        elif model_type == DetectorModelType.BERT.value.lower():
            raise NotImplementedError()
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {DetectorModelType.__name__}")

    @staticmethod
    def _build_classification_head(config):
        from src.implementation.models.components.classification_head import ClassificationHead
        return ClassificationHead(config)

    def forward(self, x):
        backbone_output = self.backbone(x)
        logits = self.classification_head(backbone_output)
        return logits

    def encode_texts(self, texts: List[str]):
        return self.text_encoder.encode_texts(texts)

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from the provided texts.
        This is useful for text encoders that require a vocabulary.
        """
        self.text_encoder.build_vocab(texts)

        # Update embedding layer if vocab size changed
        if hasattr(self.backbone, 'embedding'):
            actual_vocab_size = self.text_encoder.get_vocab_size()
            if actual_vocab_size != self.backbone.vocab_size:
                self.backbone.vocab_size = actual_vocab_size
                self.backbone.embedding = nn.Embedding(
                    actual_vocab_size,
                    self.backbone.embedding_dim,
                    padding_idx=0
                )

    def predict(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def set_output_dim_for_classification_head(self, num_classes: int):
        """
        Set the input dimension for the classification head.
        This is useful when the backbone output dimension is not known at initialization.
        """
        if self._config.num_classes != num_classes:
            logger.warning(
                f"Number of classes in config ({self._config.num_classes}) does not match provided num_classes ({num_classes})."
            )
            self.classification_head.set_output_dim(num_classes)
