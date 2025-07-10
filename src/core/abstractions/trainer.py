from abc import abstractmethod, ABC


class AbstractTrainer(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model.

        This method should be implemented by subclasses to define how the model is trained.

        @param args: Positional arguments for training.
        @param kwargs: Keyword arguments for training.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model.

        This method should be implemented by subclasses to define how the model is evaluated.

        @param args: Positional arguments for evaluation.
        @param kwargs: Keyword arguments for evaluation.
        @return: Evaluation results.
        """
        pass