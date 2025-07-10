from src.infrastructure.utils.constants import DetectorModelType as DMT

TRADITIONAL_AGENTS = [
    DMT.SVM.value,
    DMT.NAIVE_BAYES.value,
]

DEEPLEARNING_AGENTS = [

]


def get_agent(config):
    classifier_type = config.get('classifier_type', None)
    if classifier_type in TRADITIONAL_AGENTS:
        from src.implementation.agents.traditional_classifier import TraditionalClassifierAgent
        return TraditionalClassifierAgent
    elif classifier_type in DEEPLEARNING_AGENTS:
        raise NotImplementedError("Deep learning agents are not implemented yet.")
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
