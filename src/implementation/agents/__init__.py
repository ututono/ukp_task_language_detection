from src.core.abstractions.agent import AbstractAgent
from src.infrastructure.utils.constants import DetectorModelType as DMT

TRADITIONAL_AGENTS = [
    DMT.SVM.value,
    DMT.NAIVE_BAYES.value,
]

DEEPLEARNING_AGENTS = [
    DMT.CNN.value,
    DMT.LSTM.value,
    DMT.TRANSFORMER.value,
    DMT.BERT.value,
    DMT.CNN.value,
]


def get_agent(config):
    classifier_type = config.get('classifier_type', None)
    if classifier_type in TRADITIONAL_AGENTS:
        from src.implementation.agents.traditional_classifier import TraditionalClassifierAgent
        return TraditionalClassifierAgent
    elif classifier_type in DEEPLEARNING_AGENTS:
        from src.implementation.agents.deep_learning_agent import DeepLearningClassifierAgent
        return DeepLearningClassifierAgent
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
