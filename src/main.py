import logging

import hydra

from src.infrastructure.loggs.rich_utils import print_config_tree

logger = logging.getLogger(__name__)


def train_language_detection(cfg):
    from src.experiments.basic_training_experiment import TrainingDetectorExperiments
    print_config_tree(cfg)
    experiment = TrainingDetectorExperiments(cfg)
    experiment.run()


def evaluate_language_detection(cfg):
    pass


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg):
    mode = getattr(cfg, 'mode', 'train')

    if mode == 'train':
        logger.info("Starting training process...")
        train_language_detection(cfg)
    elif mode == 'evaluate':
        logger.info("Starting evaluation process...")
        evaluate_language_detection(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'evaluate'.")


if __name__ == '__main__':
    main()
