import logging
from typing import Optional

from pyexpat import features

from src.core.abstractions.agent import AbstractAgent
from src.core.abstractions.data_processor import AbstractDataProcessor
from src.core.abstractions.evaluator import AbstractEvaluator
from src.core.abstractions.trainer import AbstractTrainer
from src.implementation.evaluators.classification_evaluator import LanguageClassificationEvaluator
from src.infrastructure.utils.constants import DatasetColumns as DSC

logger = logging.getLogger(__name__)


class BasicTrainer(AbstractTrainer):
    def train(self,
              agent: AbstractAgent,
              train_data,
              val_data,
              test_data,
              vectorize_fn=None,
              evaluator: Optional[AbstractEvaluator] = None,
              *args,
              **kwargs
              ):
        # load and preprocess data
        logger.info("Starting training process...")

        # train the agent
        logger.info("Training the agent...")
        agent.build_model()

        agent.train(train_data, val_data, *args, **kwargs)

        # evaluate the agent
        logger.info("Evaluating the agent...")

        # init evaluator if not provided
        evaluator = LanguageClassificationEvaluator(
            class_labels=train_data[DSC.CLASS_LABELS]) if evaluator is None else evaluator

        predictions = agent.predict(
            test_data[DSC.TEXT],
            feature_extract_fn=vectorize_fn,
            return_labels=True
        )

        y_test_str = evaluator.convert_label_int2str(test_data[DSC.LABEL])
        results = evaluator.evaluate(predictions, y_test_str)

        logger.info("Training and evaluation completed.")

    def evaluate(
            self,
            agent: AbstractAgent,
            data,
            vectorize_fn=None,
            evaluator: Optional[AbstractEvaluator] = None,
    ):
        # init evaluator if not provided
        evaluator = LanguageClassificationEvaluator(
            class_labels=data[DSC.CLASS_LABELS]) if evaluator is None else evaluator

        predictions = agent.predict(
            data[DSC.TEXT],
            feature_extract_fn=vectorize_fn,
            return_labels=True
        )

        y_test_str = evaluator.convert_label_int2str(data[DSC.LABEL])
        results = evaluator.evaluate(predictions, y_test_str)
        logger.info("Evaluation completed.")
