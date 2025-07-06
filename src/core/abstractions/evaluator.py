from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    @abstractmethod
    def evaluate(self, predictions, ground_truth):
        """
        Evaluate the model's predictions against the ground truth.

        @param predictions: The model's predictions.
        @param ground_truth: The actual ground truth values.
        @return: Evaluation metrics or results.
        """
        pass