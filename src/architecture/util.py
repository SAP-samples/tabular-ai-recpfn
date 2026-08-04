import torch

from constants import DEVICE
from config import Config

def get_attn_mask(attn_mask:torch.Tensor|tuple, config:Config, is_causal=True) -> torch.Tensor:
    """
    Generate an attention mask based on positional embedding scheme and causality.

    :param attn_mask: Either a tuple (batch_size, num_heads, q_seq_len, k_seq_len) specifying dimensions,
                      or a tensor with shape (batch_size, num_heads, q_seq_len, k_seq_len).
    :param config: Configuration object providing positional_embedding_scheme.
    :param is_causal: Whether to use a causal mask when q_seq_len == k_seq_len.
    :return: Attention mask tensor of shape (batch_size, num_heads, q_seq_len, k_seq_len).
             For 'abs'/'nope', a boolean mask. For 'alibi', a float tensor with -inf for masked positions.
             For 'hard-alibi', a boolean mask according to the ALiBi constraints.
    """
    if isinstance(attn_mask, tuple):
        batch_size, num_heads, q_seq_len, k_seq_len = attn_mask
        if is_causal and q_seq_len == k_seq_len:
            attn_mask = get_causal_mask(q_seq_len, DEVICE).expand(batch_size, num_heads, -1, -1)
        else:
            attn_mask = torch.ones((batch_size, num_heads, q_seq_len, k_seq_len), device=DEVICE).bool()
    else:
        batch_size, num_heads, q_seq_len, k_seq_len = attn_mask.size()
    pos_emb_scheme = config.positional_embedding_scheme if config else 'abs'
    if pos_emb_scheme in ['abs', 'nope']:
        return attn_mask.bool()
    elif pos_emb_scheme in ['alibi', 'hard-alibi']:
        if k_seq_len != q_seq_len:
            return attn_mask.bool()
        else:
            seq_len = k_seq_len
        alibi = torch.arange(seq_len, device=attn_mask.device).unsqueeze(0) - torch.arange(seq_len, device=attn_mask.device).unsqueeze(1)
        alibi = alibi.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        if pos_emb_scheme == 'alibi':
            alibi = alibi.float().masked_fill(alibi > 0, float('-inf')).masked_fill(attn_mask == 0, float('-inf'))
        elif pos_emb_scheme == 'hard-alibi':
            m = torch.ones(num_heads, device=attn_mask.device) * seq_len
            m[:num_heads//2] = torch.arange(0, num_heads//2, device=attn_mask.device)
            alibi = torch.logical_and(alibi >= (-m.view(1, -1, 1, 1)),  alibi <= 0)
        return alibi

def get_causal_mask(seq_len:int, device) -> torch.Tensor:
    """
    Generate a causal mask for self-attention.
    :param seq_len: Length of the sequence.
    :param device: Device to place the mask tensor on.
    :return: Causal mask tensor of shape (1, 1, seq_len, seq_len).
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
    return mask.bool()
