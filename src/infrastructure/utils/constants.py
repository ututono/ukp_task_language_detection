# This script is to store
from enum import Enum


class TypeEnum(Enum):
    """
    Base class for enums with string representation and fast lookup.
    """

    def __init__(self, value):
        self._id = value

    def __eq__(self, other):
        if isinstance(other, TypeEnum):
            return self._id == other._id
        elif isinstance(other, str):
            return self._id == other

    def __hash__(self):
        return hash(self._id)

    def __str__(self):
        """
        Return string representation of the enum.
        @return: String representation.
        """
        return self.value


class FeatureExtractionType(TypeEnum):
    """
    Enum for feature extraction types.
    """
    TFIDF = "tf_idf"
    WORD2VEC = "word2vec"
    NGRAM = "ngram"


class DetectorModelType(TypeEnum):
    """
    Enum for detector model types.
    """
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"

class DatasetColumns(TypeEnum):
    """
    Enum for dataset column names.
    """
    TEXT = "sentence"
    LABEL = "label"
    FEATURES = "features"
    LABEL_NAMES = "label_names"
    CLASS_LABELS = "class_labels"
