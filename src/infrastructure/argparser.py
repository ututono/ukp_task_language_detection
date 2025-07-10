import argparse


class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Language Detection Training and Evaluation")

    def add_config_arguments(self):
        self.parser.add_argument("--mode", choices=["train", "evaluate"], required=True,
                                 help="Mode to run: train or evaluate")
        self.parser.add_argument("--config_path", type=str, default="../configs",
                                 help="Path to configuration files")
        self.parser.add_argument("--config_name", type=str,
                                 help="Configuration file name (without .yaml extension)")

    def add_all_arguments(self):
        """
        Add all necessary arguments for the training and evaluation script.
        This method can be extended to include more arguments as needed.
        """
        self.add_config_arguments()
        # Add more arguments here if needed

    def parse(self):
        return self.parser.parse_args()


def get_arguments():
    """
    Parse command line arguments for the training and evaluation script.

    @return: Parsed arguments
    """
    args = Arguments()
    args.add_config_arguments()
    return args.parse()
