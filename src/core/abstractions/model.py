from abc import ABC, abstractmethod
from typing import List


class AbstractModel(ABC):
    def __init__(self, config):
        """
        Initialize the model with the given configuration.

        @param config: Configuration object containing model parameters.
        """
        self._config = config
        self._model = None

    @abstractmethod
    def build(self):
        """
        Build the model based on the configuration.

        This method should be implemented by subclasses to define how the model is constructed.
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str], **kwargs) -> List[str]:
        """
        Make predictions of which language the input texts are written in.

        @param texts: List of input texts for which predictions are to be made.
        @param kwargs: Additional keyword arguments for prediction.
        @return: List of predicted scores or labels.
        """
        pass

    @classmethod
    @abstractmethod
    def build_config(cls, config):
        """
        Return config for the model, including all arguments needed to build/control the model.
        """
        pass
