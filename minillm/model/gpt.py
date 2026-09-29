"""The GPT model itself - a parameter holder, not an algorithm.

Why this file exists
--------------------
``SPEC.md`` forbids the usual pattern of a giant ``forward()`` that does
everything.  Instead this class only declares *which tensors exist*:

    wte  token embedding      (vocab -> vector)
    wpe  position embedding   (position -> vector)
    blocks  the transformer stack
    ln_f  final normalisation
    lm_head  vector -> vocabulary scores

The computation lives in :mod:`minillm.pipeline`.  ``forward`` below is
deliberately a one-line delegate so that ordinary PyTorch tooling
(``model(x)``, ``torch.compile``) keeps working while the pipeline stays the
single source of truth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import ModelConfig
from .block import TransformerBlock


class GPTModel(nn.Module):
    """A decoder-only transformer, described purely by its parameters."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Learned lookup tables: one vector per token and one per position.
        # Learned (rather than sinusoidal) positions are what GPT-2 uses.
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)

        # Projects the final hidden state to a score for every vocabulary item.
        # No bias: it would be a constant added to every token, redundant with
        # the embedding's own bias term.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        if cfg.tie_embeddings:
            # Input and output embeddings share one matrix. Saves ~19M params
            # here and usually helps a little, because "token -> vector" and
            # "vector -> token" are two sides of the same idea.
            self.lm_head.weight = self.wte.weight

    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        """Small normal init; residual output projections scaled down.

        The scaling by ``1/sqrt(2 * n_layer)`` compensates for the fact that the
        residual stream accumulates one contribution per sub-layer, which would
        otherwise grow with depth.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual_proj", False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count (useful for a sanity check before training)."""
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------ #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Thin delegate: all real work happens in :func:`minillm.pipeline.forward_pass`."""
        from ..pipeline import forward_pass  # local import avoids a circular import

        return forward_pass(self, idx, targets)
