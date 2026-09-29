"""Position-wise feed-forward network.

Why this file exists
--------------------
Attention mixes information *across* positions.  It is a weighted average, so it
is entirely linear in the values.  The feed-forward network is the part that
mixes information *within* a single position and, thanks to the non-linear
activation, gives the transformer its expressive power.  It is applied to every
token independently, which is why it is called "position-wise".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class MLP(nn.Module):
    """Parameter holder for the two-layer position-wise network.

    The hidden layer is deliberately 4x the embedding width: expand, apply a
    non-linearity in a higher-dimensional space, then project back down.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()  # smooth ReLU; standard choice since GPT-2
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # See CausalSelfAttention.c_proj: damped init for residual writers.
        self.c_proj._is_residual_proj = True
        self.dropout = nn.Dropout(cfg.dropout)


def feed_forward(x: torch.Tensor, mlp: MLP) -> torch.Tensor:
    """Run the expand -> activate -> project chain on every position."""
    x = mlp.c_fc(x)  # expand to 4 * n_embd
    x = mlp.gelu(x)  # non-linearity (the reason this layer exists)
    x = mlp.c_proj(x)  # project back to n_embd
    return mlp.dropout(x)
