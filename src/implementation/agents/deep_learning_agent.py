import logging

import torch
from datasets import ClassLabel
from torch import optim, nn
from tqdm import tqdm

from src.core.abstractions.agent import AbstractAgent
from src.core.entities.config import AgentConfig
from src.implementation.models.base_language_detector import BaseDeepLearningLanguageDetector
from src.infrastructure.utils.constants import DatasetColumns as DSC
from src.infrastructure.utils.constants import DatasetColumns

logger = logging.getLogger(__name__)


class DeepLearningClassifierAgent(AbstractAgent):

    def __init__(self, config):
        super().__init__(config)
        model_config = config.model_config
        self._model_type = model_config.get('model_type', 'CNN')
        self._lr = model_config.get('learning_rate', 0.001)
        self._batch_size = model_config.get('batch_size', 32)
        self._epochs = model_config.get('epochs', 10)
        self._embedding_dim = model_config.get('embedding_dim', 128)
        self._device = config.get('device', 'cpu')
        self._num_workers = config.get('num_workers', 8)
        self._seed = config.get('seed', 42)

        # TODO : Add support for different optimizers and loss functions
        self._optimizer = model_config.get('optimizer', 'Adam')
        self._criterion = model_config.get('loss_function', 'CrossEntropyLoss')

        self._class_labels = None

    @classmethod
    def build_config(cls, cfg):
        return AgentConfig(
            model_config=cfg.get('models', None),
            tokenizer_config=cfg.tokenizer,
            device=cfg.local.device,
            num_workers=cfg.local.num_workers,
            seed=cfg.local.seed,
            num_classes=cfg.datasets.get('num_classes', None),
            cfg=cfg,
        )

    def build_model(self):
        model_config = BaseDeepLearningLanguageDetector.build_config(self._config)
        self._model = BaseDeepLearningLanguageDetector(model_config)
        self._model.to(self._device)

        # Initialize optimizer and loss function
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._criterion = nn.CrossEntropyLoss()

        logger.info(
            f"Built {self._model_type} model with {sum(p.numel() for p in self._model.parameters())} parameters")

    def _create_dataloader(self, x, y=None):
        from torch.utils.data import DataLoader, TensorDataset
        if y is not None:
            dataset = TensorDataset(x, y)
        else:
            dataset = TensorDataset(x)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

    def create_dataloader(self, data: dict):
        texts = data.get(DSC.TEXT, None)
        labels = torch.tensor(data[DSC.LABEL], dtype=torch.long)
        label_names = data.get(DSC.LABEL_NAMES, None)
        if self._class_labels is None:
            self._class_labels = ClassLabel(names=label_names)

        self._num_classes = len(label_names)

        # Encode texts
        encoded_texts = self._model.encode_texts(texts)

        # Create a dataset and dataloader
        dataloader = self._create_dataloader(encoded_texts, labels)
        return dataloader

    def _perform_training_step(self, epoch, data_loader, optimizer, criterion, matrics):
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self._model.to(self._device)

        for batch in data_loader:
            texts, labels = batch
            texts, labels = texts.to(self._device), labels.to(self._device)

            optimizer.zero_grad()
            outputs = self._model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        matrics['train']['loss'].append(avg_loss)
        matrics['train']['accuracy'].append(accuracy)

    def _perform_validation_step(self, epoch, data_loader, criterion, matrics):
        self._model.eval()
        self._model.to(self._device)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                texts, labels = batch
                texts, labels = texts.to(self._device), labels.to(self._device)

                outputs = self._model(texts)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        matrics['val']['loss'].append(avg_loss)
        matrics['val']['accuracy'].append(accuracy)

    def train(self, train_data, val_data=None):
        if not self._model:
            raise ValueError("Model is not built. Call build_model() before training.")

        states = {
            'train': {
                'loss': [],
                'accuracy': [],
            },
            'val': {
                'loss': [],
                'accuracy': [],
            }
        }

        self._model.build_vocab(train_data[DSC.TEXT])

        train_dataloader = self.create_dataloader(train_data)
        val_dataloader = self.create_dataloader(val_data) if val_data is not None else None

        self._model.set_output_dim_for_classification_head(self._num_classes)

        logger.info(f"Starting {self._model_type} training for {self._epochs} epochs...")

        with tqdm(total=self._epochs, leave=True, position=0) as pbar:
            for epoch in range(self._epochs):
                self._perform_training_step(
                    epoch=epoch,
                    data_loader=train_dataloader,
                    optimizer=self._optimizer,
                    criterion=self._criterion,
                    matrics=states
                )

                if val_dataloader is not None:
                    self._perform_validation_step(
                        epoch=epoch,
                        data_loader=val_dataloader,
                        criterion=self._criterion,
                        matrics=states
                    )

                pbar.update(1)

                progress_msg = {
                    "train_loss": states['train']['loss'][-1],
                    "train_accuracy": states['train']['accuracy'][-1],
                    "val_loss": states['val']['loss'][-1] if val_dataloader is not None else None,
                    "val_accuracy": states['val']['accuracy'][-1] if val_dataloader is not None else None
                }
                pbar.set_postfix(**progress_msg)

        return states

    def predict(self, texts, return_labels=True, **kwargs):
        """
        Predict the class labels for the given texts.
        :param texts:
        :param return_labels: return class labels (str) if True, otherwise return raw predictions (digits)
        :param kwargs:
        :return:
        """
        encoded_texts = self._model.encode_texts(texts)
        dataloader = self._create_dataloader(encoded_texts)

        predictions = []

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                texts = batch[0].to(self._device)
                outputs = self._model(texts)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy().tolist())
        if return_labels and self._class_labels is not None:
            predictions = [self._class_labels.int2str(label) for label in predictions]

        return predictions
