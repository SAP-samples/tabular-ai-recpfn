import torch
from copy import deepcopy
import json
import logging
import os
import gc
import numpy as np
import time
from abc import abstractmethod

from constants import DEVICE
from architecture import RecPFN, EmbKNN
from dataset import SyntheticDataset, DefaultTrainDataset, DefaultTestDataset
from dataset.train import TrainDataset
from dataset.test import TestDataset
from config import Config
from train.util import combined_dataloader, check_nan
from util.metrics import MetricCollection
from util.log_time import LogTime


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for handler in logger.handlers:
    logger.removeHandler(handler)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.propagate = False

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR


class BaseTrain:
    """
    Base class for training workflows, handling dataset preparation, model loading/saving,
    and iteration logic for training/validation epochs.

    :param output_dir: Directory to store checkpoints and logs.
    :param config: Configuration object for training and model hyperparameters.
    """

    def __init__(self,
                 output_dir:str,
                 config:Config=None):
        if config is None:
            config = Config()
        self.config = config
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_train_datasets(self):
        """
        Build training and validation datasets based on configuration.

        :return: Tuple of (train_datasets, val_datasets)
        """
        if self.config.train_with_synthetic_data:
            train_datasets = [SyntheticDataset(config=self.config)]
            val_datasets = train_datasets
        else:
            train_datasets = [DefaultTrainDataset(input_dir=ds, config=self.config)
                              for ds in self.config.train_datasets]
            val_datasets = [DefaultTrainDataset(input_dir=ds, config=self.config, mode='validate')
                            for ds in self.config.train_datasets]
        return train_datasets, val_datasets

    def get_train_dataloaders(self,
                        train_datasets:list[TrainDataset],
                        val_datasets:list[TrainDataset]):
        """
        Create dataloaders for training and validation sets. Combines multiple datasets if needed.

        :param train_datasets: List of train dataset instances.
        :param val_datasets: List of validation dataset instances.
        :return: Tuple of (train_loader, val_loader)
        """
        train_loaders = [
            torch.utils.data.DataLoader(ds, self.config.batch_size, collate_fn = ds.process_batch,
                                        num_workers=0, prefetch_factor=None) for ds in train_datasets
        ]
        val_loaders = [
            torch.utils.data.DataLoader(ds, self.config.batch_size, collate_fn = ds.process_batch,
                                        num_workers=0, prefetch_factor=None) for ds in val_datasets
        ]

        if len(train_loaders) == 0:
            raise ValueError("No training datasets found.")
        elif len(train_loaders) == 1:
            train_loader = train_loaders[0]
        else:
            train_loader = combined_dataloader(train_loaders)

        if len(val_loaders) == 0:
            raise ValueError("No validation datasets found.")
        elif len(val_loaders) == 1:
            val_loader = val_loaders[0]
        else:
            val_loader = combined_dataloader(val_loaders, restart_on_exhausted=False)
        return train_loader, val_loader

    def process_train_datasets(self):
        """
        Prepare cached datasets and dataloaders for training/validation.

        :return: None
        """
        self.train_datasets, self.val_datasets = self.get_train_datasets()
        self.train_loader, self.val_loader = self.get_train_dataloaders(
            self.train_datasets, self.val_datasets)

    def load_model(self, model_path:str=None, embedding_dim:int=None):
        """
        Initialize model (and optional baseline) and load weights from a path if provided.

        :param model_path: Optional path to a saved model state_dict.
        :param embedding_dim: Embedding dimension for model initialization.
        :return: Initialized RecPFN model on DEVICE.
        """
        self.model = RecPFN(config=self.config, input_dim=embedding_dim)
        self.model = self.model.to(DEVICE)
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist.")
            logger.info(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            logger.info("Model loaded successfully.")
        if self.config.baseline_config:
            self.baseline_model = EmbKNN(config=self.config, input_dim=embedding_dim)
            self.baseline_model = self.baseline_model.to(DEVICE)
        return self.model

    def save_model(self, model_path:str):
        """
        Save current model parameters to the given filepath.

        :param model_path: Destination path for model state_dict.
        :return: None
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def iterate_one_epoch(self,
                          model: RecPFN,
                          baseline_model: EmbKNN,
                          dataloader: torch.utils.data.DataLoader,
                          optimizer: torch.optim.Optimizer = None,
                          evaluate_baseline: bool = True,
                          evaluate_last_only: bool = False,
                          scheduler_warmup=None,
                          current_step: int = 0,
                          max_num_batches: int = None
                          ):
        """
        Iterate through one epoch with optional training (optimizer provided) and evaluation.
        Supports gradient accumulation, baseline evaluation, and step-based LR scheduling.

        :param model: The primary RecPFN model.
        :param baseline_model: Baseline model (EmbKNN) for comparison metrics.
        :param dataloader: DataLoader providing batches.
        :param optimizer: Optimizer for training. If None, runs in eval mode.
        :param evaluate_baseline: Whether to compute baseline metrics.
        :param evaluate_last_only: Whether to evaluate only the last timestep.
        :param scheduler_warmup: Scheduler stepped each optimizer step (warmup or cosine).
        :param current_step: Global step offset for schedulers.
        :param max_num_batches: Optional cap on number of batches processed.
        :return: If evaluate_baseline=True, (metrics, baseline_metrics, last_step), else (metrics, last_step).
        """
        train = (optimizer is not None)
        metrics = MetricCollection(self.config)
        if evaluate_baseline:
            if isinstance(self.config.baseline_config, list):
                baseline_metrics = {mode: MetricCollection(self.config) for mode in self.config.baseline_config}
            else:
                baseline_metrics = MetricCollection(self.config)
        accumulation_steps = self.config.num_gradient_accumulation_steps if train else 1

        timers = LogTime()

        if train:
            model.train()
        else:
            model.eval()
        valid_mask = None

        timers.start_stage('dataloader')
        for batch_count, batch in enumerate(dataloader):
            timers.end_stage('dataloader')
            step = current_step + batch_count

            timers.start_stage('data_transfer')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True) if batch_count % accumulation_steps == 0 else None
            batch = tuple(t.to(DEVICE, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)
            if self.config.train_with_icl:
                input_embs, mask, mapped_labels, sample_embeddings, icl, ds_config = batch
            else:
                input_embs, mask, mapped_labels, sample_embeddings, ds_config = batch
            timers.end_stage('data_transfer')

            timers.start_stage('forward_pass')
            if self.config.train_with_icl:
                outputs = model(input_embs, mask, icl)
            else:
                outputs = model(input_embs, mask)
            timers.end_stage('forward_pass')

            timers.start_stage('loss_computation')
            loss_mask = mask
            tmp_loss, _ = \
                model.get_loss(outputs, mapped_labels, sample_embeddings,
                               mask=loss_mask, evaluate_last_only=evaluate_last_only,
                               valid_mask=valid_mask, metrics=metrics)
            timers.end_stage('loss_computation')

            if check_nan([tmp_loss, input_embs, outputs]) is not None:
                raise ValueError("NaN or Inf values detected in inputs, outputs, or loss.")

            if train:
                timers.start_stage('backward_pass')
                (tmp_loss / accumulation_steps).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                timers.end_stage('backward_pass')

                timers.start_stage('optimizer_step')
                if (batch_count + 1) % accumulation_steps == 0:
                    t0 = time.time()
                    optimizer.step()
                timers.end_stage('optimizer_step')
                if scheduler_warmup is not None:
                    scheduler_warmup.step()

            if check_nan(list(model.parameters())) is not None:
                logger.info("Model parameters contain NaN or Inf values after backward pass.")
                model.load_state_dict(torch.load(f"{self.output_dir}/model.pth"))
                if check_nan(list(model.parameters())) is not None:
                    raise ValueError("Model parameters contain NaN or Inf values even after loading checkpoint.")

            timers.start_stage('cleanup')
            del tmp_loss
            torch.cuda.empty_cache()
            timers.end_stage('cleanup')

            if evaluate_baseline:
                timers.start_stage('baseline_forward')
                baseline_outputs = baseline_model(input_embs, mask)
                timers.end_stage('baseline_forward')

                timers.start_stage('baseline_loss_computation')
                if isinstance(self.config.baseline_config, list) and isinstance(baseline_outputs, dict):
                    for mode, mode_outputs in baseline_outputs.items():
                        _, _ = \
                            baseline_model.get_loss(mode_outputs, mapped_labels,
                                                    sample_embeddings, mask=loss_mask,
                                                    evaluate_last_only=evaluate_last_only,
                                                    valid_mask=valid_mask,
                                                    metrics=baseline_metrics[mode])
                else:
                    _, _ = \
                        baseline_model.get_loss(baseline_outputs, mapped_labels,
                                                sample_embeddings, mask=loss_mask,
                                                evaluate_last_only=evaluate_last_only,
                                                valid_mask=valid_mask, metrics=baseline_metrics)
                if train:
                    torch.cuda.empty_cache()

                timers.end_stage('baseline_loss_computation')
            timers.start_stage('dataloader')
            if max_num_batches is not None and batch_count >= max_num_batches - 1:
                break

        timers.end_stage('dataloader')

        if self.config.log_time:
            logger.info(timers.log_summary())
        logger.info('Trained metrics:')
        logger.info(metrics.log_summary())
        if evaluate_baseline:
            logger.info('Baseline metrics:')
            if isinstance(self.config.baseline_config, list):
                for mode in self.config.baseline_config:
                    logger.info(f'Baseline mode: {mode}')
                    logger.info(baseline_metrics[mode].log_summary())
            else:
                logger.info(baseline_metrics.log_summary())

        del input_embs, mask, mapped_labels, sample_embeddings
        torch.cuda.empty_cache()
        if evaluate_baseline:
            return metrics, baseline_metrics, step
        else:
            return metrics, step

    @abstractmethod
    def run(self):
        """
        Abstract entrypoint to execute training loop.

        :return: Trained model instance.
        """
        pass
