import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import List, Optional
from datasets import ClassLabel

from src.core.abstractions.agent import AbstractAgent
from src.infrastructure.utils.constants import DetectorModelType, DatasetColumns as DSE

logger = logging.getLogger(__name__)


class TraditionalClassifierAgent(AbstractAgent):
    def __init__(self, config):
        """
        Initialize the traditional classifier agent with the given configuration.

        @param config: Configuration object containing agent parameters.
        """
        self._traditional_classifier_type = config.get('traditional_classifier_type', DetectorModelType.SVM)
        self._smoothing_alpha = config.get('smoothing_alpha', 1.0)
        self._svm_C = config.get('svm_C', 1.0)
        self._max_iter = config.get('max_iter', 1000)
        self._model = None
        self._class_labels:  ClassLabel = None
        super().__init__(config)

    def build_model(self):
        if self._traditional_classifier_type == DetectorModelType.SVM:
            from sklearn.svm import SVC
            self._model = SVC(C=self._svm_C, max_iter=self._max_iter, probability=True)
        elif self._traditional_classifier_type == DetectorModelType.NAIVE_BAYES:
            from sklearn.naive_bayes import MultinomialNaiveBayes
            self._model = MultinomialNaiveBayes(alpha=self._smoothing_alpha)
        else:
            raise ValueError(f"Unsupported traditional classifier type: {self._traditional_classifier_type}")

    def train(self, train_data, val_data=None):
        if self._model is None:
            self.build_model()

        X_train, y_train, label_names = train_data[DSE.FEATURES], train_data[DSE.LABEL], train_data[DSE.LABEL_NAMES]
        self._class_labels = ClassLabel(names=label_names)

        start_time = time.time()
        logger.info(f"{self._traditional_classifier_type} start training...")
        self._model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"{self._traditional_classifier_type} training completed in {elapsed_time:.2f} seconds.")

        y_train_pred = self._model.predict(X_train)
        train_results = self._compute_results(
            self._class_labels.int2str(y_train_pred),
            self._class_labels.int2str(y_train),
            label_names
        )
        logger.info(f"Train results: {train_results}")

        if val_data:
            X_val, y_val = val_data[DSE.FEATURES], val_data[DSE.LABEL]
            y_val_pred = self._model.predict(X_val)
            val_results = self._compute_results(
                self._class_labels.int2str(y_val_pred),
                self._class_labels.int2str(y_val),
                label_names
            )
            logger.info(f"Validation results: {val_results}")

    def _compute_results(self, pred, gold, label_names=None):
        from src.implementation.evaluators.classification_evaluator import LanguageClassificationEvaluator
        evaluator = LanguageClassificationEvaluator(class_labels=label_names, verbose=False)
        results = evaluator.evaluate(pred, gold)
        return results

    def evaluate(self, data):
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        X_test, y_test = data[DSE.FEATURES], data[DSE.LABEL]
        y_pred = self._model.predict(X_test)
        results = self._compute_results(y_pred, y_test, data.get('label_names'))
        return results

    def predict(self, texts: List[str], feature_extract_fn: Optional[Callable] = None, return_labels=True):
        """
        Predict the language of the given texts using the trained model.

        @param texts: List of text samples to predict.
        @param feature_extractor: Optional feature extractor to preprocess the texts.
        @return: Predicted labels for the input texts.
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        if feature_extract_fn:
            features = feature_extract_fn(texts)
        else:
            features = texts

        pred = self._model.predict(features)

        if return_labels and self._class_labels:
            return self._class_labels.int2str(pred)
        return pred

    def save_model(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{self._traditional_classifier_type.value}_model.pkl"
        import joblib
        joblib.dump(self._model, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        import joblib
        self._model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
