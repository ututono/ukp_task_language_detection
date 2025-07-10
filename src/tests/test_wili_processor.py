import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.implementation.data_processors.multilingual_processor import MultilingualDataProcessor
from src.infrastructure.utils.constants import DatasetColumns as DSC

import unittest
from unittest.mock import patch
from datasets import DatasetDict, Dataset, Features, Value, ClassLabel
from omegaconf import OmegaConf
from types import SimpleNamespace

class DummyConfig(dict):
    """Mock configuration object behaving like a dict."""
    def get(self, key, default=None):
        return self[key] if key in self else default

    def to_namespace(self):
        def convert(obj):
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: convert(v) for k, v in obj.items()})
            return obj

        return convert(self)

    def to_omegaconf(self):
        """Convert self to OmegaConf DictConfig (recursively)."""
        def convert(obj):
            if isinstance(obj, dict):
                # Recursively convert any nested DummyConfig or dict
                return {k: convert(v) for k, v in obj.items()}
            return obj
        return OmegaConf.create(convert(self))

class TestWiLiDataProcessor(unittest.TestCase):

    def setUp(self):
        """Set up the processor with default config."""
        base_dict = {
            'datasets': {
                'dataset_type': 'wili',
                'dataset_path': 'MartinThoma/wili_2018',
                'text_column': 'sentence',  # Default for WiLi dataset
                'label_column': 'label',  # Default for WiLi dataset
            },
            'method': 'traditional',
            'feature_extraction_method': 'ngram',
            'val_ratio': 0.2,
            'seed': 42,
            'ngram_range': (3, 5),
            'max_features': 10000
        }
        deep_config = {
            'datasets': {
                'dataset_type': 'wili',
                'dataset_path': 'MartinThoma/wili_2018',
                'text_column': 'sentence',  # Default for WiLi dataset
                'label_column': 'label',  # Default for WiLi dataset
            },
            "method": "deep_learning",
            "val_ratio": 0.2,
            "seed": 123
        }
        self.traditional_config = DummyConfig(base_dict)
        self.traditional_config['cfg'] = OmegaConf.create({
            **base_dict,
            'ngram_range': list(base_dict['ngram_range'])  # OmegaConf prefers list
        })
        self.traditional_config = self.traditional_config.to_omegaconf()

        self.deep_config = DummyConfig(deep_config)
        self.deep_config['cfg'] = OmegaConf.create(deep_config)
        self.deep_config = self.deep_config.to_omegaconf()

    @patch.object(MultilingualDataProcessor, "load_raw_data")
    def test_load_raw_data_mocked(self, mock_load):
        """Test loading raw data using mocked HuggingFace dataset."""
        features = Features({
            DSC.TEXT.value: Value('string'),
            DSC.LABEL.value: ClassLabel(names=["de", "en", "cn"]),
        })
        mock_data = DatasetDict({
            "train": Dataset.from_dict({"sentence": ["Text A", "Text B"], "label": [0, 1]}, features=features),
            "test": Dataset.from_dict({"sentence": ["Text C"], "label": [2]}, features=features)
        })
        mock_load.return_value = mock_data
        processor = MultilingualDataProcessor(self.traditional_config)
        data = processor.load_raw_data("MartinThoma/wili_2018")
        self.assertIn("train", data)
        self.assertIn("test", data)
        self.assertEqual(len(data["train"]), 2)

    def test_clean_data(self):
        """Test that clean_data removes whitespace and control characters."""
        processor = MultilingualDataProcessor(self.traditional_config)
        features = Features({
            DSC.TEXT.value: Value('string'),
        })
        raw = DatasetDict({
            "train": Dataset.from_dict({"sentence": ["Hello\nWorld!", "\tToo   many   spaces"]}, features=features),
            "test": Dataset.from_dict({"sentence": [" \n Strip me \t "]}, features=features)
        })
        cleaned = processor.clean_data(raw)
        self.assertEqual(cleaned["train"]["sentence"][0], "Hello World!")
        self.assertEqual(cleaned["train"]["sentence"][1], "Too many spaces")
        self.assertEqual(cleaned["test"]["sentence"][0], "Strip me")

    def test_split_data(self):
        """Test train/val split ratio and output types."""
        processor = MultilingualDataProcessor(self.traditional_config)
        features = Features({
            DSC.TEXT.value: Value('string'),
            DSC.LABEL.value: ClassLabel(names=["de", "en", "cn", "fr", "es"]),
        })
        mock_data = DatasetDict({
            "train": Dataset.from_dict({
                "sentence": [f"Sample {i}" for i in range(100)],
                "label": [i % 5 for i in range(100)]
            }, features=features),
            "test": Dataset.from_dict({
                "sentence": ["Test A", "Test B"],
                "label": [0, 1]
            }, features=features)
        })
        train, val, test = processor.split_data(mock_data)
        self.assertAlmostEqual(len(train) + len(val), 100, delta=1)
        self.assertEqual(len(test), 2)

    def test_preprocess_traditional(self):
        """Test traditional feature extraction with mock data."""
        processor = MultilingualDataProcessor(self.traditional_config)
        features = Features({
            DSC.TEXT.value: Value('string'),
            DSC.LABEL.value: ClassLabel(names=["de", "en", "cn", "fr", "es"]),
        })
        mock_dataset = Dataset.from_dict({
            DSC.TEXT.value: ["Language one.", "Language two."],
            DSC.LABEL.value: [0, 1]
        }, features=features)
        # Manually fit the vectorizer
        processor._feature_extractor.fit_vectorizer(mock_dataset["sentence"])
        result = processor.preprocess_data(mock_dataset)
        self.assertIn("sentence", result)
        self.assertEqual(result[DSC.FEATURES].shape[0], 2)
        self.assertEqual(result[DSC.LABEL], [0, 1])

    def test_preprocess_deep_learning(self):
        """Test deep learning preprocessing returns raw text and labels."""
        processor = MultilingualDataProcessor(self.deep_config)
        features = Features({
            DSC.TEXT.value: Value('string'),
            DSC.LABEL.value: ClassLabel(names=["de", "en", "cn", "fr", "es"]),
        })
        mock_dataset = Dataset.from_dict({
            "sentence": ["deep sample A", "deep sample B"],
            "label": [3, 4]
        }, features=features)
        result = processor.preprocess_data(mock_dataset)
        self.assertEqual(result["sentence"], ["deep sample A", "deep sample B"])
        self.assertEqual(result["label"], [3, 4])

    def test_validate_data(self):
        """Test that validation catches empty or invalid entries."""
        processor = MultilingualDataProcessor(self.traditional_config)
        features = Features({
            DSC.TEXT.value: Value('string'),
            DSC.LABEL.value: ClassLabel(names=["de", "en", "cn", "fr", "es"]),
        })
        # Invalid: One sentence is empty
        bad_data = DatasetDict({
            "train": Dataset.from_dict({
                "sentence": ["valid", "  "],
                "label": [0, 1]
            }, features=features)
        })
        self.assertFalse(processor.validate_data(bad_data))

        # Valid: All sentences are non-empty and labels are integers
        good_data = DatasetDict({
            "train": Dataset.from_dict({
                "sentence": ["one", "two"],
                "label": [0, 1]
            }, features=features),
            "test": Dataset.from_dict({
                "sentence": ["three"],
                "label": [2]
            }, features=features)
        })
        self.assertTrue(processor.validate_data(good_data))


if __name__ == "__main__":
    unittest.main()
