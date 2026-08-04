from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch
import os

from train.base import BaseTrain, logger
from config import Config


class Train(BaseTrain):
    """
    Concrete training runner implementing epoch loops, schedulers, validation, and early stopping.

    :param output_dir: Directory to store outputs and checkpoints.
    :param config: Configuration object for training and model hyperparameters.
    """

    def __init__(self,
                 output_dir:str,
                 config:Config=None):
        super().__init__(output_dir, config)

    def run(self, warm_start=False):
        """
        Execute the training process with warmup and cosine annealing schedulers, validation,
        checkpointing, and early stopping.

        :param warm_start: If True, load model weights from output_dir/model.pth before training.
        :return: The trained model loaded with best checkpoint.
        """
        self.process_train_datasets()
        embedding_dim = self.train_datasets[0].emb_dim

        if warm_start:
            model_path = os.path.join(self.output_dir, "model.pth")
        self.load_model(model_path if warm_start else None,
                        embedding_dim=embedding_dim)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        warmup_steps = self.config.warmup_epochs * self.config.train_epoch_size
        total_steps = self.config.num_epochs * self.config.train_epoch_size

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0
        scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)

        cosine_steps = max(1, total_steps - warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=self.config.learning_rate * 0.01)

        current_step = 0
        best_ratio = 0.0
        epochs_without_improvement = 0
        early_stopping_patience = self.config.early_stopping_patience

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Training epoch {epoch+1} (learning rate: {optimizer.param_groups[0]['lr']:.6f}):")
            _, _, _ = self.iterate_one_epoch(
                model=self.model,
                baseline_model=self.baseline_model,
                dataloader=self.train_loader,
                optimizer=optimizer,
                scheduler_warmup=scheduler_warmup if epoch * self.config.train_epoch_size < warmup_steps else scheduler_cosine,
                current_step=current_step,
                max_num_batches=self.config.train_epoch_size)
            current_step += self.config.train_epoch_size
            logger.info(f"Validation epoch {epoch+1}:")
            val_metrics, baseline_val_metrics, _ = self.iterate_one_epoch(
                model=self.model,
                baseline_model=self.baseline_model,
                dataloader=self.val_loader,
                optimizer=None,
                max_num_batches=self.config.val_epoch_size)
            early_stopping_metric = val_metrics.get(self.config.early_stopping_metric)/\
                ((baseline_val_metrics[self.config.baseline_config[0]] if isinstance(baseline_val_metrics, dict) else baseline_val_metrics)\
                 .get(self.config.early_stopping_metric) + 1e-9)
            if early_stopping_metric > best_ratio:
                best_ratio = early_stopping_metric
                epochs_without_improvement = 0
                logger.info(f"New best model found at epoch {epoch+1} with ratio: {best_ratio:.4f}")
                if self.config.save_model:
                    torch.save(self.model.state_dict(), f"{self.output_dir}/model.pth")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in validation metric for {epochs_without_improvement} epochs.")

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

            logger.info(f"Learning rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.6f}")
            torch.cuda.empty_cache()

        self.model.load_state_dict(torch.load(f"{self.output_dir}/model.pth"))
        return self.model
