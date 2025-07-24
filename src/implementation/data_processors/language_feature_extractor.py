import numpy as np
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

from src.infrastructure.utils.constants import FeatureExtractionType as FET


class LanguageFeatureExtractor:
    """
    A class for extracting features from text data for language identification.
    Supports both traditional methods (n-gram features) and deep learning methods.
    """

    def __init__(self, config):
        """
        Initialize the feature extractor with the given configuration.

        @param config: Configuration object containing feature extraction parameters.
        """
        self._config = config
        self._method = config.get('feature_extraction_method', FET.NGRAM)
        self._ngram_range = config.get('ngram_range', (1, 3))
        self._max_features = config.get('max_features', 10000)
        self._vectorizer = None

    def fit_vectorizer(self, texts: List[str]):
        """
        Fit the vectorizer on the given texts.

        @param texts: List of text samples to fit the vectorizer on.
        """
        ngram_range = tuple(self._ngram_range)
        if self._method == FET.NGRAM:
            self._vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=ngram_range,
                max_features=self._max_features
            )
        elif self._method == FET.TFIDF:
            self._vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=ngram_range,
                max_features=self._max_features
            )
        else:
            # For deep learning methods, we don't need a vectorizer
            return

        self._vectorizer.fit(texts)

    def extract_features(self, texts: List[str]) -> Union[np.ndarray, List[str]]:
        """
        Extract features from the given texts based on the configured method.

        @param texts: List of text samples to extract features from.
        @return: Feature matrix for traditional methods or preprocessed texts for deep learning methods.
        """
        if self._method in [FET.NGRAM, FET.TFIDF]:
            if self._vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
            return self._vectorizer.transform(texts)
        else:
            # For deep learning methods, we return the preprocessed texts
            return texts

    def preprocess_for_deep_learning(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for deep learning methods.

        @param texts: List of text samples to preprocess.
        @return: Preprocessed texts.
        """
        # For deep learning, we might want to do some basic preprocessing
        # like lowercasing, removing special characters, etc.
        preprocessed_texts = []
        for text in texts:
            # Basic preprocessing
            preprocessed_text = text.lower()
            # Add more preprocessing steps as needed
            preprocessed_texts.append(preprocessed_text)
        return preprocessed_texts

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features (n-grams) used by the vectorizer.

        @return: List of feature names.
        """
        if self._vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        return self._vectorizer.get_feature_names_out()