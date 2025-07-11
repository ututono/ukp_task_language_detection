import logging
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset, load_from_disk

from src.core.abstractions.cusdataset import AbstractDataset
from src.core.entities.config import DatasetConfig

logger = logging.getLogger(__name__)


class WiLiDataset(AbstractDataset):
    """
    Dataset class for the WiLi_2018 dataset from HuggingFace.
    """

    @classmethod
    def build_config(cls, cfg):
        return DatasetConfig(
            dataset_type='wili',
            dataset_path=cfg.datasets.dataset_path,
            text_column=cfg.datasets.text_column,
            label_column=cfg.datasets.label_column,
            cfg=cfg
        )

    def __init__(self, config):
        """
        Initialize the WiLi dataset with the given configuration.

        @param config: Configuration object containing dataset parameters.
        """
        super().__init__(config)
        self._path = config.get('dataset_path', 'MartinThoma/wili_2018')
        self._text_column = config.get('text_column', 'sentence')
        self._label_column = config.get('label_column', 'label')

    @property
    def text_column(self) -> str:
        """
        Get the name of the column containing the text data.

        @return: Name of the text column
        """
        return self._text_column

    @property
    def label_column(self) -> str:
        """
        Get the name of the column containing the label data.

        @return: Name of the label column
        """
        return self._label_column

    def load_data(self, path=None):
        """
        Load the WiLi_2018 dataset from HuggingFace.

        @return: Loaded dataset
        """
        path = path or self._path
        self._data = self._load_hf_data(path)
        logger.info(f'Loaded dataset from {path}')
        return self._data

    def _load_hf_data(self, path: str) -> Any:
        """
        Load dataset from Hugging Face.

        @param path: Path to the dataset on HuggingFace (e.g., "MartinThoma/wili_2018")
        @return: Loaded dataset
        """
        # Check if the path is a valid HuggingFace dataset, otherwise try to load it as a local path
        if Path(path).exists():
            return load_from_disk(path)
        return load_dataset(path)

    def preprocess(self):
        """
        Preprocess the dataset.
        For WiLi dataset, no specific preprocessing is needed at this level.
        """
        if self._data is None:
            self.load_data()
        return self._data

    def get_data(self):
        """
        Get the processed dataset.

        @return: Processed dataset ready for use.
        """
        if self._data is None:
            self.preprocess()
        return self._data
