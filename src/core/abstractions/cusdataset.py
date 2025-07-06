from abc import abstractmethod, ABC


class AbstractDataset(ABC):
    def __init__(self, config):
        """
        Initialize the dataset with the given configuration.

        @param config: Configuration object containing dataset parameters.
        """
        self._config = config
        self._data = None

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
