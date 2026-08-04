import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from constants import DEVICE
from config import Config
from architecture.util import get_attn_mask
from architecture.base import SeqRec

class SelfAttention(nn.Module):

    def __init__(self, emb_dim: int,
                 bias: bool=False,
                 is_causal: bool=False,
                 attn_type = 'A',
                 hidden_dim:int = None,
                 config: Config = None):
        """
        Initialize a SelfAttention layer.

        :param emb_dim: Embedding dimension of the input (and output).
        :param bias: Whether to include bias terms in projection layers.
        :param is_causal: If True, applies causal attention masking.
        :param attn_type: Attention variant, one of 'A' or 'B'.
        :param hidden_dim: Per-head hidden dimension; defaults to emb_dim // n_heads.
        :param config: Configuration object containing attention-related parameters.
        """
        super().__init__()

        self.config = config

        self.emb_dim = emb_dim
        self.is_causal = is_causal
        self.attn_type = attn_type
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.pos_emb_scheme = config.positional_embedding_scheme if config else 'hard-alibi'
        assert attn_type in ['A', 'B'], "attn_type must be one of 'A', 'B', or 'C'"
        self.attn_type = attn_type

        if hidden_dim is None:
            hidden_dim = emb_dim // self.n_heads
        assert emb_dim % self.n_heads == 0
        self.hidden_dim = hidden_dim

        self.k_proj = nn.Linear(emb_dim, self.n_heads*hidden_dim, bias=bias)
        self.q_proj = nn.Linear(emb_dim, self.n_heads*hidden_dim, bias=bias)

        v_hid_dim = emb_dim if attn_type=='A' else hidden_dim
        self.v_proj = nn.Linear(emb_dim, self.n_heads*v_hid_dim, bias=bias)

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weights()

        if attn_type == 'B':
            self.attn_proj = nn.Linear(self.n_heads * hidden_dim, emb_dim)
            self.attn_dropout = nn.Dropout(self.dropout)
            self.v_proj_out = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim)
            )

    def apply_ff(self, y:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        """
        Apply post-attention feed-forward and residual operations.

        :param y: Attention output tensor of shape (batch_size, n_heads, seq_len, value_dim).
        :param x: Original input tensor of shape (batch_size, seq_len, emb_dim) for residual connections.
        :return: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        b, s = x.size(0), x.size(1)
        if self.attn_type == 'B':
            y = y.transpose(1,2).contiguous().view(b, s, -1)
            y = self.attn_proj(y)
            y = self.attn_dropout(y) + x
            y = self.layer_norm(y)
            y = self.v_proj_out(y)
            y = y + x
        else:
            y = y.sum(1)/self.n_heads
            y = self.layer_norm(y) + x
        return y

    def forward(self, x:torch.Tensor, attn_mask:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the SelfAttention layer.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param attn_mask: Optional attention mask of shape (batch_size, n_heads, seq_len, seq_len)
                          or a broadcast-compatible shape for scaled dot-product attention.
        :return: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        b, s, _ = x.size()

        q:torch.Tensor = self.q_proj(x)
        k:torch.Tensor = self.k_proj(x)
        q = q.view(b, s, self.n_heads, -1).transpose(1, 2)
        k = k.view(b, s, self.n_heads, -1).transpose(1, 2)

        v:torch.Tensor = self.v_proj(x)
        v = v.view(b, s, self.n_heads, -1).transpose(1, 2)

        if self.pos_emb_scheme in ['alibi', 'hard-alibi']:
            attn_mask = get_attn_mask(attn_mask if attn_mask is not None
                                        else (b, self.n_heads, s, s),
                                      self.config, self.is_causal)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=self.is_causal if attn_mask is None else False)

        return self.apply_ff(y, x)

    def init_weights(self):
        """
        Initialize projection weights for value, key, and query.
        - For attn_type 'A', initializes value projections to approximate identity per head.
        - For attn_type 'B', initializes value projections with Xavier uniform.
        - Key and query projections are initialized with Xavier uniform; biases are zeroed.
        """
        if self.attn_type == 'A':
            self.v_proj.weight.data = torch.cat([torch.eye(self.emb_dim) for _ in range(self.n_heads)], dim=0)
            if self.v_proj.bias is not None:
                self.v_proj.bias.data = torch.zeros(self.emb_dim * self.n_heads)
        else:
            nn.init.xavier_uniform_(torch.eye(self.emb_dim))
            if self.v_proj.bias is not None:
                self.v_proj.bias.data = torch.zeros(self.n_heads * (self.emb_dim))

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        if self.k_proj.bias is not None:
            self.k_proj.bias.data = torch.zeros(self.emb_dim)
        if self.q_proj.bias is not None:
            self.q_proj.bias.data = torch.zeros(self.emb_dim)


class CrossAttention(SelfAttention):

    def __init__(self, emb_dim: int,
                 bias: bool=False,
                 is_causal: bool=False,
                 attn_type = 'A',
                 hidden_dim:int = None,
                 config: Config = None):
        """
        Initialize a CrossAttention layer.

        :param emb_dim: Embedding dimension of the input (and output).
        :param bias: Whether to include bias terms in projection layers.
        :param is_causal: If True, applies causal attention masking (unused for cross-attention).
        :param attn_type: Attention variant, one of 'A' or 'B'.
        :param hidden_dim: Per-head hidden dimension; defaults to emb_dim // n_heads.
        :param config: Configuration object containing attention-related parameters.
        """
        super().__init__(emb_dim, bias, is_causal, attn_type, hidden_dim, config)

    def forward(self, x:torch.Tensor,
                icl:torch.Tensor,
                x_mask:torch.Tensor=None,
                icl_mask:torch.Tensor=None,
                max_kv_size=100000) \
        -> torch.Tensor:
        """
        Forward pass of the CrossAttention layer.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param icl: In-context learning tensor of shape (n_icl, seq_len, emb_dim).
        :param x_mask: Optional mask for x indicating valid positions in each sequence; shape (batch_size, seq_len).
        :param icl_mask: Optional mask for icl indicating valid positions; shape (n_icl, seq_len).
        :param max_kv_size: Maximum total key/value length to control memory usage when batching queries.
        :return: Output tensor of shape (batch_size, seq_len, emb_dim).

        Note: No positional embeddings are used in this layer.
        """
        b, s, _ = x.size()
        h = self.hidden_dim
        b_icl, s_icl, _ = icl.size()

        q:torch.Tensor = self.q_proj(x)
        q = q.view(b, s, self.n_heads, -1).transpose(1, 2)

        k:torch.Tensor = self.k_proj(icl).view(1, -1, icl.shape[-1])
        k = k.view(1, -1, self.n_heads, h).transpose(1, 2)

        v:torch.Tensor = self.v_proj(icl).view(1, -1, icl.shape[-1])
        v = v.view(1, b_icl*s_icl, self.n_heads, -1).transpose(1, 2)

        if icl_mask is not None:
            v = v[:, :, icl_mask.view(-1).bool(), :]
            k = k[:, :, icl_mask.view(-1).bool(), :]

        if not v.shape[-2]:
            return x

        max_seqs_per_batch = max_kv_size // v.shape[-2]
        y = []
        for i in range(0, b, max_seqs_per_batch):
            end = min(i + max_seqs_per_batch, b)
            q_ = q[i:end,:,:,:]
            mask_ = x_mask[i:end,:] if x_mask is not None else None
            attn_mask = None
            k_ = k
            v_ = v

            tmp = F.scaled_dot_product_attention(
                q_, k_, v_, dropout_p=self.dropout if self.training else 0.0,
                is_causal=False, attn_mask=attn_mask)

            tmp = self.apply_ff(tmp, x[i:end,:,:])
            y.append(tmp)

        y = torch.cat(y, dim=0)
        return y

class RecPFN(SeqRec):
    def __init__(self, input_dim:int, config:Config, device=None):
        """
        Initialize the RecPFN model.

        :param input_dim: Embedding dimension of inputs (and model hidden size).
        :param config: Configuration object with model hyperparameters.
        :param device: Torch device to place model parameters on.
        """
        super(RecPFN, self).__init__(input_dim, config, device)

        self.icl_transformer = None

        if self.config.icl_module_type == 'A':
            cross_attn_type = ['A'] * self.config.n_layers
            self_attn_type = ['A'] * self.config.n_layers
        elif self.config.icl_module_type == 'B':
            cross_attn_type = ['B'] * self.config.n_layers
            self_attn_type = ['B'] * self.config.n_layers
        elif self.config.icl_module_type == 'AB':
            cross_attn_type = ['A' if i < self.config.n_layers // 2 else 'B' for i in range(self.config.n_layers)]
            self_attn_type = ['A' if i < self.config.n_layers // 2 else 'B' for i in range(self.config.n_layers)]
        elif self.config.icl_module_type == 'alternating':
            modes = ['A', 'B']
            cross_attn_type = [modes[i%2] for i in range(self.config.n_layers)]
            self_attn_type = [modes[i%2] for i in range(self.config.n_layers)]

        self.cross_decoders = nn.ModuleList([
            CrossAttention(emb_dim=input_dim,
                           hidden_dim=input_dim // config.n_heads,
                           attn_type=cross_attn_type[i],
                           config=config).to(self.device)
            for i in range(config.n_layers)
        ])
        self.encoders = nn.ModuleList([
            SelfAttention(emb_dim=input_dim,
                          hidden_dim=input_dim // config.n_heads,
                          attn_type=self_attn_type[i],
                          is_causal=True,
                          config=self.config).to(self.device)
            for i in range(config.n_layers)
        ])

        self.out_norm = nn.LayerNorm(self.input_dim, elementwise_affine=False).to(device)

        if self.config.positional_embedding_scheme == 'abs':
            self.positional_embedding = nn.Embedding(config.max_sequence_len, input_dim).to(self.device)

    def forward(self, x:torch.Tensor, mask:torch.Tensor, icl:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RecPFN model.
        :param x: Input tensor of shape (batch_size, seq_len, emb_dim).
        :param mask: Mask tensor of shape (batch_size, seq_len) indicating valid positions.
        :param icl: In-context learning tensor of shape (n_examples, seq_len, emb_dim).
        :return: Output tensor of shape (batch_size, seq_len, emb_dim).
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input embedding dimension {x.shape[-1]} does not match model input dimension {self.input_dim}.")
        if self.config.positional_embedding_scheme == 'abs':
            positions = torch.arange(x.size(1), device=self.device).unsqueeze(0).expand_as(x[:,:,0])
            pos_emb = self.positional_embedding(positions)
            x = x + pos_emb
        x = x.to(self.device)
        if icl is not None:
            icl = icl.to(self.device)
            icl_mask = icl.abs().sum(dim=-1).not_equal(0)

        for i in range(self.config.n_layers):
            if icl is not None:
                icl = self.encoders[i](icl)
                icl = icl * icl_mask.unsqueeze(-1).float()
            x = self.encoders[i](x)
            x = x * mask.unsqueeze(-1).float()
            if icl is not None:
                x = self.cross_decoders[i](x, icl, x_mask=mask, icl_mask=icl_mask)
                x = x * mask.unsqueeze(-1).float()
        x = self.out_norm(x)
        x = x * mask.unsqueeze(-1).float()
        return x
