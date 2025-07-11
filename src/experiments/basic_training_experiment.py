import logging

from src.core.abstractions.agent import AbstractAgent
from src.core.abstractions.data_processor import AbstractDataProcessor
from src.core.abstractions.trainer import AbstractTrainer
from src.implementation.agents import get_agent
from src.implementation.data_processors import get_processor
from src.implementation.evaluators import get_evaluator
from src.implementation.trainers import get_trainer
from src.infrastructure.utils.constants import DatasetColumns as DSE
from src.infrastructure.utils.general import set_random_seed

logger = logging.getLogger(__name__)


class TrainingDetectorExperiments:

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        """
        Placeholder method to run the language detection experiment.
        """
        set_random_seed(self.cfg.local.seed)

        logger.info(f"Start loading data...")
        ProcessorClass = get_processor()
        process_config = ProcessorClass.build_config(cfg=self.cfg)

        processor: AbstractDataProcessor = ProcessorClass(process_config)
        train_data, val_data, test_data = processor.process_pipeline()

        logger.info(f"Data loaded and processed successfully. Train size: {len(train_data[DSE.TEXT])}, Validation size: {len(val_data[DSE.TEXT])}, Test size: {len(test_data[DSE.TEXT])}")
        TrainerClass = get_trainer()
        trainer: AbstractTrainer = TrainerClass()

        AgentClass = get_agent(self.cfg.models)
        agent_config = AgentClass.build_config(cfg=self.cfg)
        agent: AbstractAgent = AgentClass(agent_config)

        EvaluatorClass = get_evaluator()
        evaluator = EvaluatorClass(class_labels=train_data[DSE.CLASS_LABELS])

        trainer.train(
            agent=agent,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            vectorize_fn=processor.vectorize_data,
            evaluator=evaluator
        )

        logger.info("Training and evaluation completed.")