import logging

import torch
import torch.nn as nn
from typing import List, Dict, Union
from collections import Counter
import re

from src.core.entities.config import TextEncoderConfig

logger = logging.getLogger(__name__)


class TextEncoder:
    """Unified text encoder supporting multiple backends"""

    def __init__(self, config: TextEncoderConfig):
        if config.get('encoder_type', None) == 'custom':
            self.encoder = CustomTextEncoder(config)
        elif config.get('encoder_type',  None) == 'huggingface':
            self.encoder = HuggingFaceTextEncoder(config)
        else:
            raise ValueError(f"Unsupported encoder type: {config.get('encoder_type', None)}")

    @classmethod
    def build_config(cls, config: Dict[str, Union[str, int]]) -> TextEncoderConfig:
        """Build configuration for TextEncoder"""
        return TextEncoderConfig(
            encoder_type=config.get('encoder_type', 'custom'),
            encoding_type=config.get('encoding_type', 'char'),
            vocab_size=config.get('vocab_size', 1000),
            max_length=config.get('max_length', 200),
            model_name=config.get('model_name', 'bert-base-multilingual-cased'),
            use_fast_tokenizer=config.get('use_fast_tokenizer', True),
            cfg=config.get('cfg', None)
        )

    def build_vocab(self, texts: List[str]) -> int:
        """Build vocabulary from training texts"""
        return self.encoder.build_vocab(texts)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to numerical sequences"""
        return self.encoder.encode_texts(texts)

    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return self.encoder.get_vocab_size()


class HuggingFaceTextEncoder:
    """HuggingFace transformers-based text encoder"""

    def __init__(self, config: TextEncoderConfig):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers library not found. Install with: pip install transformers")

        # Configuration
        self.model_name = config.get('model_name', 'bert-base-multilingual-cased')
        self.use_fast = config.get('use_fast_tokenizer', True)
        self.max_length = config.get('max_length', 200)

        # Initialize tokenizer
        self.vocab_built = True  # Pretrained tokenizers have built vocabularies
        self._init_tokenizer()

        logger.info(f"Initialized HuggingFace encoder with {self.model_name}")

    def _init_tokenizer(self):
        """Initialize the HuggingFace tokenizer"""
        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast,
                model_max_length=self.max_length
            )

            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, falling back to bert-base-multilingual-cased")
            self.model_name = 'bert-base-multilingual-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=self.use_fast,
                model_max_length=self.max_length
            )

    def build_vocab(self, texts: List[str]) -> int:
        """Build vocabulary (no-op for pretrained tokenizers)"""
        self.vocab_built = True
        logger.info(f"Using pretrained vocabulary with {len(self.tokenizer)} tokens")
        return len(self.tokenizer)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using HuggingFace tokenizer"""
        if not self.vocab_built:
            logger.warning("Vocabulary not built, building automatically")
            self.build_vocab(texts)

        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded['input_ids']

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)


class CustomTextEncoder:
    """Unified text encoder supporting both character and word level encoding"""

    def __init__(self, config):
        self.encoding_type = config.get('encoding_type', 'char')
        self.vocab_size = config.get('vocab_size', 1000)
        self.max_length = config.get('max_length', 200)

        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.special_tokens = [self.pad_token, self.unk_token]

        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_built = False

    @classmethod
    def build_config(cls, config: Dict[str, Union[str, int]]) -> TextEncoderConfig:
        """Build configuration for TextEncoder"""
        return TextEncoderConfig(
            encoding_type=config.get('encoding_type', None),
            vocab_size=config.get('vocab_size', None),
            max_length=config.get('max_length', None),
            cfg=config.get('cfg', None)
        )

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        if self.encoding_type == 'char':
            return self._build_char_vocab(texts)
        else:
            return self._build_word_vocab(texts)

    def _build_char_vocab(self, texts: List[str]):
        """Build character-level vocabulary"""
        char_counts = Counter()

        for text in texts:
            # Clean and normalize text
            text = self._clean_text(text)
            char_counts.update(text)

        # Get most frequent characters
        most_common = char_counts.most_common(self.vocab_size - len(self.special_tokens))

        # Build mappings
        self.char_to_idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx_to_char = {idx: token for idx, token in enumerate(self.special_tokens)}

        for idx, (char, _) in enumerate(most_common):
            real_idx = idx + len(self.special_tokens)
            self.char_to_idx[char] = real_idx
            self.idx_to_char[real_idx] = char

        self.vocab_built = True
        return len(self.char_to_idx)

    def _build_word_vocab(self, texts: List[str]):
        """Build word-level vocabulary"""
        word_counts = Counter()

        for text in texts:
            text = self._clean_text(text)
            words = text.split()
            word_counts.update(words)

        # Get most frequent words
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))

        # Build mappings
        self.char_to_idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx_to_char = {idx: token for idx, token in enumerate(self.special_tokens)}

        for idx, (word, _) in enumerate(most_common):
            real_idx = idx + len(self.special_tokens)
            self.char_to_idx[word] = real_idx
            self.idx_to_char[real_idx] = word

        self.vocab_built = True
        return len(self.char_to_idx)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Convert texts to numerical sequences"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        encoded_sequences = []

        for text in texts:
            text = self._clean_text(text)

            if self.encoding_type == 'char':
                sequence = self._encode_chars(text)
            else:
                sequence = self._encode_words(text)

            encoded_sequences.append(sequence)

        # Pad sequences to max_length
        return self._pad_sequences(encoded_sequences)

    def _encode_chars(self, text: str) -> List[int]:
        """Encode text at character level"""
        sequence = []
        for char in text[:self.max_length]:
            idx = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
            sequence.append(idx)
        return sequence

    def _encode_words(self, text: str) -> List[int]:
        """Encode text at word level"""
        words = text.split()[:self.max_length]
        sequence = []
        for word in words:
            idx = self.char_to_idx.get(word, self.char_to_idx[self.unk_token])
            sequence.append(idx)
        return sequence

    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """Pad sequences to uniform length"""
        padded = []
        pad_idx = self.char_to_idx[self.pad_token]

        for seq in sequences:
            if len(seq) < self.max_length:
                seq.extend([pad_idx] * (self.max_length - len(seq)))
            else:
                seq = seq[:self.max_length]
            padded.append(seq)

        return torch.tensor(padded, dtype=torch.long)

    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return len(self.char_to_idx) if self.vocab_built else 0
