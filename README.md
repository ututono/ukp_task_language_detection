# ukp_task_language_detection
## Introduction

A modular language detection benchmark supporting both traditional machine learning and deep learning approaches for multilingual text classification.

Features include:
- Multiple Algorithms: Support for SVM, Naive Bayes, CNN, LSTM, and Transformer models, and potential for pre-trained models.
- Feature Extraction: N-gram and TF-IDF features for traditional methods
- Deep Learning: Character/word-level encoders with neural architectures
- WiLi Dataset: Built-in support for the WiLi-2018 dataset (235 languages)
- Configurable Pipeline: Hydra-based configuration management and MLflow integration for experiment tracking
- Comprehensive Evaluation: Detailed metrics and confusion matrix visualization

Github repository: [ukp-task-language-detection](https://github.com/ututono/ukp_task_language_detection/tree/dev)

### Project Structure
```
src/
├── core/                    # Abstract interfaces
│   ├── abstractions/        # ABC classes for extensibility
│   └── entities/           # Configuration dataclasses
├── implementation/         # Concrete implementations
│   ├── agents/             # Model training agents
│   ├── data_processors/    # Data loading and preprocessing
│   ├── evaluators/         # Evaluation metrics
│   ├── models/             # Neural network architectures
│   └── trainers/           # Training orchestration
└── infrastructure/         # Utilities and constants
│   └── loggs/              # Utils for logging
│   └── utils/              # Genearl utilities
├── config/                 # Configuration files


## Setup
### Install the required packages:

```bash
conda create -n lang_detect python=3.11
conda activate lang_detect
pip install -r requirements.txt
```
Note: If you encounter dependency conflicts, you can try installing the `requirements_mini.txt` file instead, which contains a minimal set of dependencies.

### Environment Variables
Set the following environment variables in your `.env` file or directly in your shell by `export`ing them:

```bash
PROJECT_ROOT=<path_to_project_root>
```

## Usage
### Training pipeline
You can simply run the training pipeline using the following command, which will use the default configuration:

#### Training for traditional machine learning models
```bash
python src/main.py \
    --config-name "training" \
    models="svm" \
    data_processor="traditional" \
    
```

#### Training for deep learning models
```bash
python src/main.py \
    --config-name "training" \
    models="lstm" \
    data_processor="deep_learning" \
```

To configure the training pipeline, you can modify configuration files of certain components in the `config` directory. For example, to change the model parameters, you can modify the `config/models/lstm.yaml` file.

## Configuration

```text
.
└── configs/
    ├── agents/
    ├── data_processor/        # config for data pipline, e.g., vocab size
    ├── datasets/              # dataset name
    ├── exp_logger/            # model logger e.g., mlflow
    ├── extras/                # for developing
    ├── hydra/
    ├── local/                 # define global configuration,e.g., seed and device
    ├── models/                # define model customized configuration
    ├── paths/            
    ├── training.yaml        # entrance for training
    └── evaluating.yaml      # entrance for evaluating
```

### Supported Models
**Traditional classifier**
- SVM
- Naive Bayes

**Deep learning classifier**
- CNN
- LSTM
- Transformer

