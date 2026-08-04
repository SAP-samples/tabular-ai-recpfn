import torch

from architecture.base import SeqRec
from config import Config

class EmbKNN(SeqRec):

    def __init__(self, input_dim:int, config:Config, device=None):
        """
        Initialize the embedding-based KNN baseline.
        :param input_dim: Dimensionality of the input embeddings.
        :param config: Configuration with baseline settings.
        :param device: Optional device to run the model on. Defaults to constants.DEVICE.
        """
        super(EmbKNN, self).__init__(input_dim, config, device)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """
        Forward pass for the EmbKNN baseline.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param mask: Mask tensor indicating valid positions in the sequence.
        :return: Aggregated embeddings according to the configured mode(s), with mask applied.
        """
        modes = self.config.baseline_config
        apply_mask = lambda t: t * mask.unsqueeze(-1).float()
        if isinstance(modes, list):
            return {mode: apply_mask(self._apply_mode(x, mode)) for mode in modes}
        return apply_mask(self._apply_mode(x, modes))

    def _apply_mode(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Apply the selected baseline mode to aggregate embeddings.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param mode: Baseline mode name (e.g., 'balanced', 'recent').
        :return: Aggregated tensor with the same shape as x.
        """
        if mode == 'balanced':
            return self.normalized_cumsum(x, dim=1)
        if mode == 'recent':
            decay = self.config.baseline_config.get('recent_decay', 0.9)
            return self.weighted_cumsum(x, dim=1, decay=decay)
        raise ValueError(f"Unknown baseline mode: {mode}")

    def normalized_cumsum(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Compute normalized cumulative sum along a dimension (simple cumulative mean).
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param dim: Dimension along which to compute the cumulative sum.
        :return: Tensor where each position is the cumulative mean up to that position.
        """
        size_dim = x.size(dim)
        shape = [1 if i != dim else size_dim for i in range(x.dim())]
        norm = torch.arange(1, size_dim + 1, device=self.device).view(shape)
        x = x.cumsum(dim=dim) / norm
        return x

    def weighted_cumsum(self, x: torch.Tensor, dim: int = 1, decay: float = 0.9) -> torch.Tensor:
        """
        Compute a weighted cumulative mean favoring recent interactions via EMA:
        y_t = (1 - beta) * x_t + beta * y_{t-1}, where beta=decay.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param dim: Sequence dimension (default 1).
        :param decay: EMA decay in [0,1). Higher gives longer memory; lower favors recent more.
        :return: Tensor where each position is the EMA up to that position.
        """
        if dim != 1:
            x = x.transpose(dim, 1)
        beta = decay
        one_minus_beta = 1.0 - beta

        b, s, d = x.shape
        y = torch.empty_like(x)
        y[:, 0, :] = x[:, 0, :]
        for t in range(1, s):
            y[:, t, :] = one_minus_beta * x[:, t, :] + beta * y[:, t - 1, :]

        if dim != 1:
            y = y.transpose(1, dim)
        return y
