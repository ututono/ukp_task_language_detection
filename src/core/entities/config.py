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




@dataclass
class DatasetConfig(ConfigBase):
    """
    Configuration class for datasets, defining parameters for dataset loading and processing.
    """
    dataset_type: str = 'wili'
    dataset_path: Optional[str] = None
    text_column: str = 'sentence'
    label_column: str = 'label'

    cfg: Optional[dict] = None



@dataclass
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



