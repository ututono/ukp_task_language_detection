from abc import abstractmethod, ABC, abstractproperty


class AbstractDataset(ABC):
    def __init__(self, config):
        """
        Initialize the dataset with the given configuration.

        @param config: Configuration object containing dataset parameters.
        """
        self._config = config
        self._data = None

    @property
    @abstractmethod
    def text_column(self) -> str:
        """
        Get the name of the column containing the text data.

        @return: Name of the text column
        """
        pass

    @property
    @abstractmethod
    def label_column(self) -> str:
        """
        Get the name of the column containing the label data.

        @return: Name of the label column
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Load the dataset based on the configuration.

        This method should be implemented by subclasses to define how the dataset is loaded.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Preprocess the dataset.

        This method should be implemented by subclasses to define how the dataset is preprocessed.
        """
        pass

    @abstractmethod
    def get_data(self):
        """
        Get the processed dataset.

        @return: Processed dataset ready for use.
        """
        pass

    @classmethod
    @abstractmethod
    def build_config(cls, cfg):
        """
        Build the configuration for the dataset based on the provided configuration object.

        @param cfg: Configuration object containing dataset parameters.
        @return: Configuration object for the dataset.
        """
        pass
