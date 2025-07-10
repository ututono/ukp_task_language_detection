import hydra

from src.infrastructure.argparser import get_arguments
from src.infrastructure.loggs.rich_utils import print_config_tree


def train_language_detection(cfg):
    from src.experiments.basic_training_experiment import TrainingDetectorExperiments
    print_config_tree(cfg)
    experiment = TrainingDetectorExperiments(cfg)
    experiment.run()


def evaluate_language_detection(cfg):
    pass


def main():
    args = get_arguments()

    if args.mode == "train":
        config_name = args.config_name or "training"
        with hydra.initialize(config_path=args.config_path, version_base=None):
            cfg = hydra.compose(config_name=config_name)
            train_language_detection(cfg)

    elif args.mode == "evaluate":
        config_name = args.config_name or "evaluating"
        with hydra.initialize(config_path=args.config_path, version_base=None):
            cfg = hydra.compose(config_name=config_name)
            evaluate_language_detection(cfg)


if __name__ == '__main__':
    main()
