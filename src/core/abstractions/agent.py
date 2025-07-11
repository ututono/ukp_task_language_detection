from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self, config):
        """
        Initialize the agent with the given configuration.

        @param config: Configuration object containing agent parameters.
        """
        self._config = config
        self._model = None

    @abstractmethod
    def build_model(self):
        """
        Build the model based on the configuration.
        :return:
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, texts, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def build_config(cls, cfg):
        """
        Build the configuration for the agent based on the provided config.

        @param cfg: Configuration object containing agent parameters.
        @return: Configured parameters for the agent.
        """
        pass
