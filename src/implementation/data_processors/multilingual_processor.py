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
from src.implementation.data_processors.language_feature_extractor import LanguageFeatureExtractor
from src.infrastructure.utils.constants import DatasetColumns as DSC

class WiLiDataProcessor(AbstractDataProcessor):
    """
    Data processor for the WiLi_2018 dataset from HuggingFace.
    Supports both traditional methods (n-gram features + Naive Bayes/SVM) and deep learning methods.
    """

    def __init__(self, config):
        """
        Initialize the WiLi data processor with the given configuration.

        @param config: Configuration object containing data processing parameters.
        """
        super().__init__(config)
        self._feature_extractor = LanguageFeatureExtractor(config)
        self._method = config.get('method', 'traditional')  # 'traditional' or 'deep_learning'
        self._val_ratio = config.get('val_ratio', 0.15)
        self._seed = config.get('seed', 42)

    def _load_hf_data(self, path: str) -> Any:
        """ 
        Load dataset from Hugging Face.

        @param path: Path to the dataset on HuggingFace (e.g., "MartinThoma/wili_2018")
        @return: Loaded dataset
        """
        from datasets import load_dataset, load_from_disk
        # Check if the path is a valid HuggingFace dataset, otherwise try to load it as a local path
        if Path(path).exists():
            return load_from_disk(path)
        return load_dataset(path)

    def load_raw_data(self, path: str):
        """
        Load the raw WiLi_2018 dataset from HuggingFace.

        @param path: Path to the dataset on HuggingFace (e.g., "MartinThoma/wili_2018")
        @return: Loaded dataset
        """
        return self._load_hf_data(path)

    def clean_data(self, data):
        def clean(text):
            text = re.sub(r"[\r\n\t]+", " ", text)  # remove line breaks, tabs
            text = re.sub(r"\s+", " ", text)        # collapse multiple spaces
            return text.strip()

        for split in data:
            data[split] = data[split].map(lambda ex: {"sentence": clean(ex["sentence"])})
        return data

    def split_data(self, data):
        train_data = data["train"]

        # initialize the feature extractor on the training data
        if self._method == "traditional":
            self._feature_extractor.fit_vectorizer(train_data["sentence"])

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
        if self._method == "traditional":
            # Extract features using the fitted vectorizer
            features = self._feature_extractor.extract_features(data["sentence"])
            labels = data["label"]
            return {
                DSC.TEXT.value: data['sentence'],
                DSC.LABEL.value: labels,
                DSC.LABEL_NAMES.value: data.features["label"].names,
                DSC.CLASS_LABELS.value: data.features["label"],
                DSC.FEATURES.value: features,
            }
        else:
            # TODO For deep learning methods
            return {
                DSC.TEXT.value: data['sentence'],
                DSC.LABEL.value: data['label'],
                DSC.LABEL_NAMES.value: data.features["label"].names,
                DSC.CLASS_LABELS.value: data.features["label"],
                DSC.FEATURES.value: None,
            }

    def validate_data(self, data):
        """
        Check for empty text or invalid labels.
        """

        def check_example(example):
            is_valid = bool(example["sentence"].strip()) and isinstance(example["label"], int)
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
