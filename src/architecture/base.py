import torch
from torch import nn
import abc

from config import Config
from util.metrics import MetricCollection
from constants import DEVICE

class SeqRec(torch.nn.Module):

    def __init__(self,
                 input_dim:int,
                 config:Config,
                 device=None):
        """
        Initialize the sequential recommender base class.
        :param input_dim: Dimensionality of the input embeddings.
        :param config: Configuration object with model and training parameters.
        :param device: Optional device to run the model on. Defaults to constants.DEVICE.
        """
        super(SeqRec, self).__init__()
        self.device = device if device is not None else DEVICE
        self.config = config
        self.input_dim = input_dim


    @abc.abstractmethod
    def forward(self, x:torch.Tensor, mask:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, seq, emb_dim).
        :param mask: Mask tensor indicating valid positions in the sequence.
        """
        x = None
        return x

    @staticmethod
    def last_nonzero_position_indices(mask:torch.Tensor) -> torch.Tensor:
        """
        Get indices of the last non-zero positions for each sequence in the batch.
        :param mask: Mask tensor of shape (batch_size, seq_len) where non-zero indicates valid positions.
        :return: Tensor of shape (batch_size,) with the index of the last non-zero position per sequence.
        """
        nonzero_mask = mask != 0
        indices = torch.argsort(nonzero_mask, dim=1, descending=True, stable=True)
        last_nonzero_indices = torch.gather(indices, 1, (nonzero_mask.sum(dim=1, keepdim=True) - 1).clamp(min=0))
        return last_nonzero_indices.squeeze()

    def get_loss(self, outputs:torch.Tensor, targets:torch.Tensor, embeddings:torch.Tensor,
                 mask:torch.Tensor=None, evaluate_last_only=False, valid_mask:torch.Tensor=None,
                 metrics:MetricCollection=None):
        """
        Compute the loss between outputs and targets.
        :param outputs: Model outputs (batch_size, seq_len, emb_dim)
        :param targets: Ground truth targets (batch_size, seq_len)
        :param embeddings: Embeddings for computing softmax cross entropy loss
        :param mask: Mask tensor indicating valid positions in the sequence
        :param evaluate_last_only: Whether to evaluate only the last position in the sequence
        :param valid_mask: Optional mask to filter valid items in evaluation
        :return: Loss tensor and metrics dictionary.
        """
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)
        embeddings = embeddings.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        if evaluate_last_only:
            last_nonzero = self.last_nonzero_position_indices(mask)
            outputs = outputs[torch.arange(outputs.shape[0]), last_nonzero, :].view(-1, outputs.shape[-1])
            x = targets[torch.arange(outputs.shape[0]), torch.clamp(last_nonzero-1,0)].view(-1)
            x[last_nonzero==0] = 0
            targets = targets[torch.arange(outputs.shape[0]), last_nonzero].view(-1)
        elif mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

        logits = torch.matmul(outputs, embeddings.T)
        if valid_mask is not None:
            logits[:, ~valid_mask] = -1e9
        if self.config.loss == 'mse':
            target_embs = embeddings[targets]
            loss = torch.nn.functional.mse_loss(outputs, target_embs, reduction='mean')
        elif self.config.loss == 'sce':
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss}")

        if self.config.optimizer !='adamw':
            reg_loss = sum(torch.norm(param, p=2) for param in self.parameters())
            size_reg_loss = torch.norm(outputs, p=2)
            loss += self.config.reg_lambda * (reg_loss + size_reg_loss)

        scores, preds = torch.topk(logits, k=metrics.max_k, dim=-1)
        metrics.update(preds, targets, loss.item())

        return loss, metrics

    def get_prediction(self, outputs:torch.Tensor, targets:torch.Tensor, embeddings:torch.Tensor, k:int):
        """
        Get the top-k predictions from the model outputs.
        :param outputs: Model outputs (batch_size, seq_len, emb_dim)
        :param targets: Ground truth targets (batch_size, seq_len)
        :param embeddings: Embeddings for computing softmax cross entropy loss
        :param k: Number of top predictions to return.
        :return: Top-k predictions and their scores
        """
        logits = torch.matmul(outputs, embeddings.T)
        scores, preds = torch.topk(logits, k=k, dim=-1)
        return preds, scores
