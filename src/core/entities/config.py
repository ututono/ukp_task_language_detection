from dataclasses import asdict

from typing import Optional, Tuple

from dataclasses import dataclass


@dataclass
class ConfigBase:
    # Convert the dataclass to a dictionary
    def to_dict(self):
        return asdict(self)

    def __init__(self, **kwargs):
        """Needs to explicitly redefine the __init__ method in every subclass of ConfigBase in order to add undefined fields to the instance."""

        # Get all attributes defined in the class
        defined_attrs = set(dir(self.__class__))

        # Apply defined fields to the instance
        for key, value in kwargs.items():
            # Set the attribute, regardless of whether it's defined
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self):
        pass

    def get(self, key, default=None):
        return getattr(self, key, default)


class DatasetConfig(ConfigBase):
    """
    Configuration class for datasets, defining parameters for dataset loading and processing.
    """
    dataset_type: str = 'wili'
    dataset_path: Optional[str] = None
    text_column: str = 'sentence'
    label_column: str = 'label'

    cfg: Optional[dict] = None


class DataProcessorConfig(ConfigBase):
    """
    Configuration class for data processors, defining parameters for data processing tasks.
    """
    method: str = 'traditional'
    val_ratio: float = 0.15
    seed: int = 42
    feature_extraction_method: str = 'ngram'
    ngram_range: Tuple[int, int] = (3, 5)
    max_features: int = 10000

    cfg: Optional[dict] = None


class AgentConfig(ConfigBase):
    model_config: Optional[dict] = None
    tokenizer_config: Optional[dict] = None
    use_sampling: bool = False
    max_samples: int = 10000
    device: str = 'cpu'
    num_workers: int = 4
    seed: int = 42
    num_classes: int = 235
    verbose = True
    cfg: Optional[dict] = None


class DLModelConfig(ConfigBase):
    """
    Configuration class for deep learning models, defining parameters for model architecture and training.
    """
    model_type: str = 'CNN'
    num_classes: int = 235
    backbone_config: Optional[dict] = None
    classification_head_config: Optional[dict] = None
    text_encoder_config: Optional[dict] = None
    device: str = 'cpu'
    backbone_output_dim: int = 128  # Redefine this after backbone initialization
    cfg: Optional[dict] = None


class TextEncoderConfig(ConfigBase):
    """
    Configuration class for text encoders, defining parameters for text encoding.
    """
    encoding_type: str = 'char'
    vocab_size: int = 10000
    max_length: int = 256
