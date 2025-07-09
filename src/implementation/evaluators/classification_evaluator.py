import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import ClassLabel
from src.core.abstractions.evaluator import AbstractEvaluator

from datasets import ClassLabel

logger = logging.getLogger(__name__)


class LanguageClassificationEvaluator(AbstractEvaluator):
    def __init__(self, class_labels: Optional[ClassLabel | List[str]] = None, verbose: bool = True):
        """
        :param class_labels: Optional list of language codes, sorted
        :param verbose: If True, print classification report
        """
        self.class_labels = class_labels if isinstance(class_labels, ClassLabel) else ClassLabel(
            names=class_labels) if class_labels else None
        self.verbose = verbose
        self.results = {}
        self._label_names: List[str] = self.class_labels.names if self.class_labels else None

    def convert_label_int2str(self, labels: List[int]) -> List[str]:
        if self.class_labels:
            return [self.class_labels.int2str(label) for label in labels]
        elif self._label_names:
            logger.warning("No ClassLabel provided, using label names directly. There may be inconsistencies.")
            return [self._label_names[label] for label in labels]
        else:
            raise ValueError("No label names provided.")

    def evaluate(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """
        Compute standard classification metrics and confusion matrix
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")

        acc = accuracy_score(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=self._label_names, zero_division=0, average=None
        )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average="macro", zero_division=0
        )
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average="micro", zero_division=0
        )

        cm = confusion_matrix(ground_truth, predictions, labels=self._label_names)

        report_df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }, index=self._label_names)

        if self.verbose:
            print("\nAccuracy:", acc)
            print("\nClassification Report:\n", report_df.round(4))

        self.results = {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "report": report_df,
            "confusion_matrix": cm,
            "labels": self._label_names,
            "total_samples": len(predictions),
        }
        return self.results

    def plot_confusion_matrix(self, figsize=(12, 10), save_path=None):
        if not self.results or "confusion_matrix" not in self.results:
            raise ValueError("No confusion matrix found. Run evaluate() first.")

        cm = self.results["confusion_matrix"]
        labels = self.results["labels"]

        plt.figure(figsize=figsize)
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", square=True, fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Language Detection Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def get_misclassified_examples(
            self,
            predictions: List[str],
            ground_truth: List[str],
            texts: List[str],
            max_per_pair: int = 10
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Collect misclassified examples for error analysis.

        :param predictions: List of predicted language codes
        :param ground_truth: List of ground-truth language codes
        :param texts: Original text samples
        :param max_per_pair: Maximum examples per (true, predicted) pair
        :return: Dictionary keyed by "true→pred" with list of examples
        """
        assert len(predictions) == len(ground_truth) == len(texts), "Length mismatch in inputs"

        errors = defaultdict(list)

        for pred, true, text in zip(predictions, ground_truth, texts):
            if pred != true:
                key = f"{true}→{pred}"
                if len(errors[key]) < max_per_pair:
                    errors[key].append({
                        "text": text[:300] + "..." if len(text) > 300 else text,
                        "true": true,
                        "pred": pred,
                        "length": len(text)
                    })

        return dict(errors)

    def print_evaluation_report(self):
        """
        Log a comprehensive evaluation report.
        """
        if not self.results:
            logger.warning("No evaluation results available. Run evaluate() first.")
            return

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LANGUAGE DETECTION EVALUATION REPORT")
        report_lines.append("=" * 60)

        # Overall metrics
        report_lines.append("\nOVERALL METRICS:")
        report_lines.append(f"Accuracy: {self.results['accuracy']:.4f}")
        report_lines.append(f"Macro F1-Score: {self.results['macro_f1']:.4f}")
        report_lines.append(f"Micro F1-Score: {self.results['micro_f1']:.4f}")
        report_lines.append(f"Total Samples: {self.results['total_samples']}")

        # Per-language metrics
        report_lines.append("\nPER-LANGUAGE METRICS:")
        report_lines.append(f"{'Language':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        report_lines.append("-" * 60)

        # TODO add evaluation metric for each language
        # for lang in self.results['languages']:
        #     precision = self.results['precision_per_language'][lang]
        #     recall = self.results['recall_per_language'][lang]
        #     f1 = self.results['f1_per_language'][lang]
        #     support = self.results['support_per_language'].get(lang, 0)
        #     report_lines.append(f"{lang:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")

        report_lines.append("-" * 60)
        report_lines.append(f"{'macro avg':<12} {self.results['macro_precision']:<12.4f} "
                            f"{self.results['macro_recall']:<12.4f} {self.results['macro_f1']:<12.4f} "
                            f"{self.results['total_samples']:<12}")
        report_lines.append(f"{'micro avg':<12} {self.results['micro_precision']:<12.4f} "
                            f"{self.results['micro_recall']:<12.4f} {self.results['micro_f1']:<12.4f} "
                            f"{self.results['total_samples']:<12}")

        # Combine all lines into a single logger statement
        full_report = "\n".join(report_lines)
        logger.info("\n" + full_report)
