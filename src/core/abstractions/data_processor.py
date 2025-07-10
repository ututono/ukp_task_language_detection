from abc import abstractmethod, ABC
from os import PathLike
from typing import Optional


class AbstractDataProcessor(ABC):

    def __init__(self, config):
        """
        Initialize the data processor with the given configuration.

        @param config: Configuration object containing data processing parameters.
        """
        self._config = config

    @abstractmethod
    def load_raw_data(self, path):
        """
        Load the raw data based on the configuration.

        This method should be implemented by subclasses to define how the raw data is loaded.
        """
        pass

    @abstractmethod
    def clean_data(self, data):
        """
        Clean the raw data.

        This method should be implemented by subclasses to define how the raw data is cleaned.

        @param data: Raw data to be cleaned.
        @return: Cleaned data.
        """
        pass

    @abstractmethod
    def split_data(self, data):
        """
        Split the cleaned data into training, validation, and test sets.

        This method should be implemented by subclasses to define how the data is split.

        @param data: Cleaned data to be split.
        @return: Tuple containing training, validation, and test sets.
        """
        pass

    @abstractmethod
    def preprocess_data(self, data):
        """
        Preprocess the data for model training.

        This method should be implemented by subclasses to define how the data is preprocessed.

        @param data: Data to be preprocessed.
        @return: Preprocessed data ready for model training.
        """
        pass

    @abstractmethod
    def validate_data(self, data):
        """
        Validate the data to ensure it meets the required standards.

        This method can be overridden by subclasses to implement specific validation logic.

        @param data: Data to be validated.
        @return: Boolean indicating whether the data is valid.
        """
        # Default implementation can be empty or raise NotImplementedError
        return True

    def process_pipeline(self, path: Optional[PathLike|str] = None):
        """
        Process the data pipeline from loading raw data to preprocessing it.

        This method orchestrates the entire data processing pipeline.

        @param path: Path to the raw data file.
        @return: Preprocessed data ready for model training.
        """
        raw_data = self.load_raw_data(path)
        cleaned_data = self.clean_data(raw_data)
        is_valid = self.validate_data(raw_data)
        if not is_valid:
            raise ValueError("Data validation failed. Please check the raw data.")

        train_data, val_data, test_data = self.split_data(cleaned_data)
        preprocessed_train_data = self.preprocess_data(train_data)
        preprocessed_val_data = self.preprocess_data(val_data)
        preprocessed_test_data = self.preprocess_data(test_data)

        return preprocessed_train_data, preprocessed_val_data, preprocessed_test_data

    def vectorize_data(self, data):
        """
        Vectorize the data using the feature extraction method defined in the configuration.
        """
        return data

    @property
    def data_text_column(self) -> str:
        """
        Get the text column name from the dataset.

        @return: Text column name
        """
        return self._dataset.text_column if hasattr(self, '_dataset') else self._config.get('text_column', None)

    @property
    def data_label_column(self) -> str:
        """
        Get the label column name from the dataset.

        @return: Label column name
        """
        return self._dataset.label_column if hasattr(self, '_dataset') else self._config.get('label_column', None)

    @classmethod
    @abstractmethod
    def build_config(cls, **kwargs):
        """
        Build a configuration object for the data processor from the hydra configuration.

        This method should be implemented by subclasses to define how the configuration is built.

        @param kwargs: Additional parameters for building the configuration.
        @return: Configuration object.
        """
        pass