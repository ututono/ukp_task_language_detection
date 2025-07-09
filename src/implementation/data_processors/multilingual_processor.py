import os
import json
import random
from pathlib import Path

import numpy as np
from typing import List, Tuple, Dict, Any, Union
from collections import defaultdict, Counter
import pickle
import re
from sklearn.model_selection import train_test_split

from src.core.abstractions.data_processor import AbstractDataProcessor
from src.core.abstractions.cusdataset import AbstractDataset
from src.implementation.data_processors.language_feature_extractor import LanguageFeatureExtractor
from src.infrastructure.utils.constants import DatasetColumns as DSC
from src.implementation.datasets.wili_dataset import WiLiDataset


class MultilingualDataProcessor(AbstractDataProcessor):
    """
    Generic data processor for multilingual datasets.
    Supports both traditional methods (n-gram features + Naive Bayes/SVM) and deep learning methods.
    """

    def __init__(self, config):
        """
        Initialize the multilingual data processor with the given configuration.

        @param config: Configuration object containing data processing parameters.
        """
        super().__init__(config)
        self._feature_extractor = LanguageFeatureExtractor(config)
        self._method = config.get('method', 'traditional')  # 'traditional' or 'deep_learning'
        self._val_ratio = config.get('val_ratio', 0.15)
        self._seed = config.get('seed', 42)
        self._dataset = None
        self._init_dataset()

    def _init_dataset(self):
        dataset_type = self._config.get('dataset_type', 'wili')
        if dataset_type == 'wili':
            self._dataset = WiLiDataset(self._config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


    def load_raw_data(self, path: str):
        """
        Load the raw dataset.

        @param path: Path to the dataset
        @return: Loaded dataset
        """
        return self._dataset.load_data()

    def clean_data(self, data):
        """
        Clean the raw data.

        @param data: Raw data to be cleaned.
        @return: Cleaned data.
        """
        def clean(text):
            text = re.sub(r"[\r\n\t]+", " ", text)  # remove line breaks, tabs
            text = re.sub(r"\s+", " ", text)        # collapse multiple spaces
            return text.strip()

        # Get the text column name from the dataset
        text_column = self._dataset.text_column

        for split in data:
            data[split] = data[split].map(lambda ex: {text_column: clean(ex[text_column])})
        return data

    def split_data(self, data):
        """
        Split the cleaned data into training, validation, and test sets.

        @param data: Cleaned data to be split.
        @return: Tuple containing training, validation, and test sets.
        """
        train_data = data["train"]
        text_column = self._dataset.text_column

        # initialize the feature extractor on the training data
        if self._method == "traditional":
            self._feature_extractor.fit_vectorizer(train_data[text_column])

        temp = train_data.train_test_split(
            test_size=self._val_ratio,
            seed=self._seed,
        )
        train_split, val_split = temp["train"], temp["test"]
        test_data = data["test"]

        return train_split, val_split, test_data

    def preprocess_data(self, data):
        """
        Preprocess the data for model training.

        For traditional methods, extracts n-gram features.
        For deep learning methods, performs basic text preprocessing.

        @param data: Data to be preprocessed
        @return: Preprocessed data ready for model training
        """
        text_column = self._dataset.text_column
        label_column = self._dataset.label_column

        if self._method == "traditional":
            # Extract features using the fitted vectorizer
            features = self._feature_extractor.extract_features(data[text_column])
            labels = data[label_column]
            return {
                DSC.TEXT.value: data[text_column],
                DSC.LABEL.value: labels,
                DSC.LABEL_NAMES.value: data.features[label_column].names,
                DSC.CLASS_LABELS.value: data.features[label_column],
                DSC.FEATURES.value: features,
            }
        else:
            # For deep learning methods
            return {
                DSC.TEXT.value: data[text_column],
                DSC.LABEL.value: data[label_column],
                DSC.LABEL_NAMES.value: data.features[label_column].names,
                DSC.CLASS_LABELS.value: data.features[label_column],
                DSC.FEATURES.value: None,
            }

    def validate_data(self, data):
        """
        Check for empty text or invalid labels.

        @param data: Data to be validated.
        @return: Boolean indicating whether the data is valid.
        """
        text_column = self._dataset.text_column
        label_column = self._dataset.label_column

        def check_example(example):
            is_valid = bool(example[text_column].strip()) and isinstance(example[label_column], int)
            return {"is_valid": is_valid}

        for split in data:
            result = data[split].map(check_example, remove_columns=data[split].column_names)
            if not all(result["is_valid"]):
                return False

        return True

    def vectorize_data(self, data: List[str]):
        """
        Vectorize the data using the feature extractor.

        @param data: Data to be vectorized
        @return: Vectorized data
        """
        if self._method == "traditional":
            return self._feature_extractor.extract_features(data)
        else:
            # For deep learning methods, we return the raw text
            return data


class WiLiDataProcessor(MultilingualDataProcessor):
    """
    Data processor for the WiLi_2018 dataset from HuggingFace.
    This class is maintained for backward compatibility.
    It is a specialized version of MultilingualDataProcessor configured for the WiLi dataset.
    """

    def __init__(self, config):
        """
        Initialize the WiLi data processor with the given configuration.

        @param config: Configuration object containing data processing parameters.
        """
        # Ensure dataset_type is set to 'wili' and use default column names for backward compatibility
        config_copy = dict(config)
        config_copy['dataset_type'] = 'wili'
        config_copy['text_column'] = config.get('text_column', 'sentence')
        config_copy['label_column'] = config.get('label_column', 'label')
        super().__init__(config_copy)

    def _load_hf_data(self, path: str) -> Any:
        """
        Load dataset from Hugging Face.
        Maintained for backward compatibility.

        @param path: Path to the dataset on HuggingFace (e.g., "MartinThoma/wili_2018")
        @return: Loaded dataset
        """
        if self._dataset is None:
            self._dataset = WiLiDataset(self._config)
        return self._dataset._load_hf_data(path)
