"""Multi-head causal self-attention - parameters and maths kept separate.

Composition note
----------------
:class:`CausalSelfAttention` **only holds parameters**.  The actual computation
lives in the free function :func:`causal_self_attention`.  This is the central
idea of the project (see ``SPEC.md``): a module is a bag of tensors that can be
passed around, while the pipeline decides what maths to run and in which order.

Why attention works this way
----------------------------
Each token builds three vectors: a *query* ("what am I looking for?"), a *key*
("what do I contain?") and a *value* ("what do I pass on if I'm relevant?").
Compatibility between a query and every preceding key decides how much of each
value flows into the token.  The mask enforces *causality*: token *t* may only
see tokens ``<= t``, otherwise the model could cheat by reading the answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Parameter holder for one multi-head causal attention layer."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim

        # One fused projection produces Q, K and V at once. A single matmul is
        # faster than three separate ones that would read the same input thrice.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Mixes the heads back into a single residual stream vector.
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # Flag read by GPTModel._init_weights: this projection writes directly
        # into the residual stream, so its initial scale is damped by depth.
        self.c_proj._is_residual_proj = True

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular boolean mask: True = "may attend", False = "forbidden".
        # Registered as a buffer so it follows the module to GPU but is not a
        # trainable parameter. `persistent=False` keeps it out of checkpoints.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)


def causal_self_attention(x: torch.Tensor, attn: CausalSelfAttention, cfg: ModelConfig) -> torch.Tensor:
    """Apply multi-head causal attention to a sequence of token vectors.

    Parameters
    ----------
    x
        Activations of shape ``(batch, time, n_embd)``.
    attn
        The parameter holder created from the same ``cfg``.

    Returns
    -------
    Tensor of shape ``(batch, time, n_embd)`` - one updated vector per token.
    """
    batch, time, channels = x.shape

    # 1. Project once and split into queries, keys and values.
    query, key, value = attn.c_attn(x).split(channels, dim=2)

    # 2. Split the channel dimension into heads so each head can specialise.
    #    (B, T, C) -> (B, n_head, T, head_dim)
    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, time, attn.n_head, attn.head_dim).transpose(1, 2)

    query, key, value = split_heads(query), split_heads(key), split_heads(value)

    # 3. Score every query against every key. Scaling by sqrt(head_dim) keeps
    #    the dot products from growing with width, which would saturate softmax.
    scores = (query @ key.transpose(-2, -1)) / (attn.head_dim**0.5)  # (B, H, T, T)

    # 4. Forbid looking into the future.
    scores = scores.masked_fill(~attn.mask[:, :, :time, :time], float("-inf"))

    # 5. Normalise into a probability distribution over previous tokens.
    weights = F.softmax(scores, dim=-1)
    weights = attn.attn_dropout(weights)

    # 6. Weighted sum of the values, then merge the heads back together.
    out = weights @ value  # (B, H, T, head_dim)
    out = out.transpose(1, 2).contiguous().view(batch, time, channels)

    # 7. Final linear mix + regularisation before the residual add (which the
    #    caller performs).
    return attn.resid_dropout(attn.c_proj(out))
