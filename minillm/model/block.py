"""One transformer block: attention and feed-forward wired with residuals.

Why this file exists
--------------------
A transformer is "just" this block repeated ``n_layer`` times.  Extracting it
means the model definition (``gpt.py``) reads like a table of contents, and the
number of layers is a config value rather than duplicated code.

Design choices
--------------
* **Pre-norm** (``LayerNorm`` *before* each sub-layer, as in GPT-2) is more
  stable to train than post-norm, so it tolerates a larger learning rate.
* **Residual connections** (``x = x + f(x)``) give gradients a direct path to
  the earlier layers; without them deep stacks simply do not train.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention import CausalSelfAttention, causal_self_attention
from .mlp import MLP, feed_forward


class TransformerBlock(nn.Module):
    """Parameter holder grouping the two sub-layers of a transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)


def transformer_block_forward(x: torch.Tensor, block: TransformerBlock, cfg: ModelConfig) -> torch.Tensor:
    """Communicate across positions, then think within each position.

    Order matters: normalise -> sub-layer -> add back to the residual stream.
    """
    x = x + causal_self_attention(block.ln_1(x), block.attn, cfg)
    x = x + feed_forward(block.ln_2(x), block.mlp)
    return x
