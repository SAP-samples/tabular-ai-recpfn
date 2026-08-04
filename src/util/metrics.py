from abc import ABC, abstractmethod
from typing import List, Union
import torch

from config import Config


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self):
        self.reset()

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Batch update

        :param args: Positional arguments for the metric-specific update.
        :param kwargs: Keyword arguments for the metric-specific update.
        :return: None
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the final metric value.

        :return: The computed metric value.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset metric state.

        :return: None
        """
        pass


class HitRate(Metric):
    """Hit Rate @ K metric."""

    def __init__(self, k: int = 10):
        self.k = k
        super().__init__()

    def reset(self):
        self.hits = 0
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update with batch predictions and labels.

        :param preds: Tensor of predicted item ids with shape (batch, num_pred), ordered by relevance.
        :param labels: Tensor of true item ids with shape (batch, 1) or (batch,).
        :return: None
        """
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        topk = preds[:, :self.k]
        hits_batch = (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        self.hits += int(hits_batch)
        self.total += preds.size(0)

    def compute(self):
        return self.hits / self.total if self.total > 0 else 0.0


class MRR(Metric):
    """Mean Reciprocal Rank metric."""

    def __init__(self, k: int = 10):
        """
        Initialize MRR@K metric.

        :param k: Cutoff for MRR (compute within top-k).
        :return: None
        """
        self.k = k
        super().__init__()

    def reset(self):
        self.reciprocal_ranks = 0.0
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update with batch predictions and labels.

        :param preds: Tensor of predicted item ids with shape (batch, num_pred), ordered by relevance.
        :param labels: Tensor of true item ids with shape (batch, 1) or (batch,).
        :return: None
        """
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        topk = preds[:, :self.k]
        match = topk == labels.unsqueeze(1)
        pos = torch.where(match, torch.arange(1, self.k + 1).unsqueeze(0).expand_as(topk), torch.zeros_like(topk))
        ranks = pos.clone()
        ranks[ranks == 0] = self.k + 1
        min_ranks = ranks.min(dim=1).values
        rr = torch.where(min_ranks <= self.k, 1.0 / min_ranks.float(), torch.zeros_like(min_ranks, dtype=torch.float))
        self.reciprocal_ranks += rr.sum().item()
        self.total += preds.size(0)

    def compute(self):
        return self.reciprocal_ranks / self.total if self.total > 0 else 0.0


class Loss(Metric):
    """Average loss metric."""

    def __init__(self):
        super().__init__()

    def reset(self):
        self.sum_loss = 0.0
        self.total = 0

    def update_single(self, loss: float):
        self.sum_loss += loss
        self.total += 1

    def update(self, losses: Union[List[float], torch.Tensor], batch_size: int = None):
        if isinstance(losses, torch.Tensor):
            if losses.numel() == 1:
                self.sum_loss += losses.item()
                self.total += batch_size if batch_size else 1
            else:
                losses = losses.cpu().numpy()
                for loss in losses:
                    self.update_single(float(loss))
        elif isinstance(losses, (list, tuple)):
            for loss in losses:
                self.update_single(loss)
        else:
            self.sum_loss += float(losses)
            self.total += batch_size if batch_size else 1

    def compute(self):
        return self.sum_loss / self.total if self.total > 0 else 0.0


class MetricCollection:
    """Collection of multiple metrics."""

    def __init__(self, config:Config):
        self.metrics = {}
        for metric_name in config.metrics:
            for k in config.eval_k:
                key = f"{metric_name}@{k}"
                if metric_name == 'hr':
                    self.metrics[key] = HitRate(k)
                elif metric_name == 'mrr':
                    self.metrics[key] = MRR(k)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
        self.metrics['loss'] = Loss()
        self.max_k = max(config.eval_k)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float = None):
        for metric in self.metrics.values():
            if isinstance(metric, Loss) and loss is not None:
                metric.update(loss)
            else:
                metric.update(preds, labels)

    def to_dict(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def get(self, metric_name: str):
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found in collection.")
        return self.metrics[metric_name].compute()

    def log_summary(self):
        log_str = ""
        for name, metric in self.metrics.items():
            log_str += f"{name}: {metric.compute():.4f};  "
        return log_str
