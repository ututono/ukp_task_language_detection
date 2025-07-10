import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from datasets import ClassLabel

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.implementation.evaluators.classification_evaluator import LanguageClassificationEvaluator


class TestLanguageClassificationEvaluator(unittest.TestCase):
    """Test suite for LanguageClassificationEvaluator metric calculations"""

    def setUp(self):
        """Setup test fixtures before each test method"""
        self.class_labels = ClassLabel(names=['en', 'zh', 'fr'])
        self.evaluator = LanguageClassificationEvaluator(
            class_labels=self.class_labels,
            verbose=False
        )

    def test_perfect_prediction_metrics(self):
        """Test metrics calculation when all predictions are correct"""
        # Test data: all predictions match ground truth
        ground_truth = ['en', 'zh', 'fr', 'en', 'zh', 'fr']
        predictions = ['en', 'zh', 'fr', 'en', 'zh', 'fr']

        results = self.evaluator.evaluate(predictions, ground_truth)

        # All metrics should be 1.0 for perfect predictions
        self.assertEqual(results['accuracy'], 1.0)
        self.assertEqual(results['macro_precision'], 1.0)
        self.assertEqual(results['macro_recall'], 1.0)
        self.assertEqual(results['macro_f1'], 1.0)
        self.assertEqual(results['micro_precision'], 1.0)
        self.assertEqual(results['micro_recall'], 1.0)
        self.assertEqual(results['micro_f1'], 1.0)

        # Check per-class metrics
        report_df = results['report']
        self.assertTrue(all(report_df['precision'] == 1.0))
        self.assertTrue(all(report_df['recall'] == 1.0))
        self.assertTrue(all(report_df['f1'] == 1.0))

        # Check confusion matrix is diagonal
        cm = results['confusion_matrix']
        expected_cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        np.testing.assert_array_equal(cm, expected_cm)

    def test_simple_three_class_metrics(self):
        """Test metrics calculation with a simple 3-class scenario"""
        # Test data with known expected results
        ground_truth = ['en', 'en', 'zh', 'zh', 'fr', 'fr']
        predictions = ['en', 'zh', 'zh', 'zh', 'fr', 'en']

        results = self.evaluator.evaluate(predictions, ground_truth)

        # Manual calculation:
        # Accuracy = 4/6 = 0.6667
        # en: TP=1, FP=1, FN=1 -> P=1/2=0.5, R=1/2=0.5, F1=0.5
        # zh: TP=2, FP=1, FN=0 -> P=2/3≈0.667, R=2/2=1.0, F1=0.8
        # fr: TP=1, FP=0, FN=1 -> P=1/1=1.0, R=1/2=0.5, F1≈0.667

        self.assertAlmostEqual(results['accuracy'], 4 / 6, places=10)

        # Check per-class metrics
        report_df = results['report']

        # English metrics
        self.assertAlmostEqual(report_df.loc['en', 'precision'], 0.5, places=10)
        self.assertAlmostEqual(report_df.loc['en', 'recall'], 0.5, places=10)
        self.assertAlmostEqual(report_df.loc['en', 'f1'], 0.5, places=10)

        # Chinese metrics
        self.assertAlmostEqual(report_df.loc['zh', 'precision'], 2 / 3, places=10)
        self.assertAlmostEqual(report_df.loc['zh', 'recall'], 1.0, places=10)
        self.assertAlmostEqual(report_df.loc['zh', 'f1'], 0.8, places=10)

        # French metrics
        self.assertAlmostEqual(report_df.loc['fr', 'precision'], 1.0, places=10)
        self.assertAlmostEqual(report_df.loc['fr', 'recall'], 0.5, places=10)
        self.assertAlmostEqual(report_df.loc['fr', 'f1'], 2 / 3, places=10)

        # Check support values
        self.assertEqual(report_df.loc['en', 'support'], 2)
        self.assertEqual(report_df.loc['zh', 'support'], 2)
        self.assertEqual(report_df.loc['fr', 'support'], 2)

    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation and structure"""
        ground_truth = ['en', 'en', 'zh', 'zh', 'fr', 'fr']
        predictions = ['en', 'zh', 'zh', 'zh', 'fr', 'en']

        results = self.evaluator.evaluate(predictions, ground_truth)
        cm = results['confusion_matrix']

        # Expected confusion matrix:
        # Rows: true labels (en, zh, fr)
        # Cols: predicted labels (en, zh, fr)
        # en->en: 1, en->zh: 1, en->fr: 0
        # zh->en: 0, zh->zh: 2, zh->fr: 0
        # fr->en: 1, fr->zh: 0, fr->fr: 1
        expected_cm = np.array([
            [1, 1, 0],  # en true
            [0, 2, 0],  # zh true
            [1, 0, 1]  # fr true
        ])

        np.testing.assert_array_equal(cm, expected_cm)

        # Verify row sums equal support for each class
        report_df = results['report']
        for i, label in enumerate(results['labels']):
            self.assertEqual(cm[i].sum(), report_df.loc[label, 'support'])

    def test_macro_vs_micro_averages(self):
        """Test macro vs micro average calculations with imbalanced data"""
        # Create imbalanced dataset
        ground_truth = ['en'] * 10 + ['zh'] * 2 + ['fr'] * 1
        predictions = ['en'] * 8 + ['zh'] * 2 + ['zh'] * 2 + ['fr'] * 1

        results = self.evaluator.evaluate(predictions, ground_truth)

        # Manual verification of micro vs macro averages
        # Micro averages should weight by support
        # Macro averages should be simple arithmetic mean

        # Calculate expected micro averages (should equal accuracy for multiclass)
        total_correct = 8 + 2 + 1  # en: 8/10, zh: 2/2, fr: 1/1
        total_samples = 13
        expected_micro_avg = total_correct / total_samples

        self.assertAlmostEqual(results['micro_precision'], expected_micro_avg, places=10)
        self.assertAlmostEqual(results['micro_recall'], expected_micro_avg, places=10)
        self.assertAlmostEqual(results['micro_f1'], expected_micro_avg, places=10)
        self.assertAlmostEqual(results['accuracy'], expected_micro_avg, places=10)

        # Macro averages should be different from micro due to imbalance
        self.assertGreater(abs(results['macro_precision'] - results['micro_precision']), 1e-6)
        self.assertGreater(abs(results['macro_recall'] - results['micro_recall']), 1e-6)
        self.assertGreater(abs(results['macro_f1'] - results['micro_f1']), 1e-6)

    def test_sklearn_consistency(self):
        """Test that our calculations match sklearn's direct calculations"""
        ground_truth = ['en', 'en', 'zh', 'zh', 'fr', 'fr', 'en']
        predictions = ['en', 'zh', 'zh', 'zh', 'fr', 'en', 'fr']

        # Calculate using our evaluator
        results = self.evaluator.evaluate(predictions, ground_truth)

        # Calculate using sklearn directly
        labels = ['en', 'zh', 'fr']
        sklearn_accuracy = accuracy_score(ground_truth, predictions)
        sklearn_precision, sklearn_recall, sklearn_f1, sklearn_support = \
            precision_recall_fscore_support(ground_truth, predictions, labels=labels, zero_division=0, average=None)

        sklearn_macro_p, sklearn_macro_r, sklearn_macro_f1, _ = \
            precision_recall_fscore_support(ground_truth, predictions, average="macro", zero_division=0)

        sklearn_micro_p, sklearn_micro_r, sklearn_micro_f1, _ = \
            precision_recall_fscore_support(ground_truth, predictions, average="micro", zero_division=0)

        sklearn_cm = confusion_matrix(ground_truth, predictions, labels=labels)

        # Compare results
        self.assertAlmostEqual(results['accuracy'], sklearn_accuracy, places=10)
        self.assertAlmostEqual(results['macro_precision'], sklearn_macro_p, places=10)
        self.assertAlmostEqual(results['macro_recall'], sklearn_macro_r, places=10)
        self.assertAlmostEqual(results['macro_f1'], sklearn_macro_f1, places=10)
        self.assertAlmostEqual(results['micro_precision'], sklearn_micro_p, places=10)
        self.assertAlmostEqual(results['micro_recall'], sklearn_micro_r, places=10)
        self.assertAlmostEqual(results['micro_f1'], sklearn_micro_f1, places=10)

        # Compare per-class metrics
        report_df = results['report']
        for i, label in enumerate(labels):
            self.assertAlmostEqual(report_df.loc[label, 'precision'], sklearn_precision[i], places=10)
            self.assertAlmostEqual(report_df.loc[label, 'recall'], sklearn_recall[i], places=10)
            self.assertAlmostEqual(report_df.loc[label, 'f1'], sklearn_f1[i], places=10)
            self.assertAlmostEqual(report_df.loc[label, 'support'], sklearn_support[i], places=10)

        # Compare confusion matrix
        np.testing.assert_array_equal(results['confusion_matrix'], sklearn_cm)

    def test_zero_division_handling(self):
        """Test handling of zero division cases (no predictions for a class)"""
        # Create scenario where one class has no predictions
        ground_truth = ['en', 'en', 'zh', 'zh', 'fr', 'fr']
        predictions = ['en', 'en', 'zh', 'zh', 'en', 'en']  # No 'fr' predictions

        results = self.evaluator.evaluate(predictions, ground_truth)

        # French should have precision=0 (no predictions), recall=0 (no correct predictions)
        report_df = results['report']
        self.assertEqual(report_df.loc['fr', 'precision'], 0.0)
        self.assertEqual(report_df.loc['fr', 'recall'], 0.0)
        self.assertEqual(report_df.loc['fr', 'f1'], 0.0)
        self.assertEqual(report_df.loc['fr', 'support'], 2)

        # Other classes should have non-zero metrics
        self.assertGreater(report_df.loc['en', 'precision'], 0)
        self.assertGreater(report_df.loc['zh', 'precision'], 0)

    def test_input_validation(self):
        """Test input validation for mismatched lengths"""
        ground_truth = ['en', 'zh', 'fr']
        predictions = ['en', 'zh']  # Different length

        with self.assertRaises(ValueError) as context:
            self.evaluator.evaluate(predictions, ground_truth)

        self.assertIn("Predictions and ground truth must have the same length", str(context.exception))

    def test_single_class_scenario(self):
        """Test metrics when only one class is present"""
        ground_truth = ['en', 'en', 'en', 'en']
        predictions = ['en', 'en', 'zh', 'zh']  # 50% accuracy

        results = self.evaluator.evaluate(predictions, ground_truth)

        self.assertEqual(results['accuracy'], 0.5)

        # Check that results contain all expected classes
        report_df = results['report']
        self.assertIn('en', report_df.index)
        self.assertIn('zh', report_df.index)
        self.assertIn('fr', report_df.index)

        # English should have recall=0.5 (2 out of 4 correct)
        self.assertEqual(report_df.loc['en', 'recall'], 0.5)
        self.assertEqual(report_df.loc['en', 'support'], 4)

        # Chinese should have precision=0 (predicted but not in ground truth)
        self.assertEqual(report_df.loc['zh', 'precision'], 0.0)
        self.assertEqual(report_df.loc['zh', 'support'], 0)

    def test_results_structure(self):
        """Test the structure and completeness of results dictionary"""
        ground_truth = ['en', 'zh', 'fr']
        predictions = ['en', 'zh', 'fr']

        results = self.evaluator.evaluate(predictions, ground_truth)

        # Check all expected keys are present
        expected_keys = {
            'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'micro_precision', 'micro_recall', 'micro_f1', 'report',
            'confusion_matrix', 'labels', 'total_samples'
        }

        self.assertEqual(set(results.keys()), expected_keys)

        # Check data types
        self.assertIsInstance(results['accuracy'], (float, np.float64))
        self.assertIsInstance(results['report'], pd.DataFrame)
        self.assertIsInstance(results['confusion_matrix'], np.ndarray)
        self.assertIsInstance(results['labels'], list)
        self.assertIsInstance(results['total_samples'], int)

        # Check DataFrame structure
        report_df = results['report']
        expected_columns = {'precision', 'recall', 'f1', 'support'}
        self.assertEqual(set(report_df.columns), expected_columns)
        self.assertEqual(list(report_df.index), ['en', 'zh', 'fr'])

        # Check confusion matrix shape
        cm = results['confusion_matrix']
        self.assertEqual(cm.shape, (3, 3))

        self.assertEqual(results['total_samples'], 3)


if __name__ == '__main__':
    unittest.main()