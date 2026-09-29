"""The forward pass, expressed as a pipeline of small independent steps.

Why this file exists
--------------------
This is the heart of the "composition instead of inheritance" goal in
``SPEC.md``.  Rather than one monolithic ``forward()`` method, the pass is a
short, readable chain of functions.  Each step *does one job*, takes the pieces
it needs explicitly, and can be tested, reordered or replaced in isolation.

    tokens -> embed_tokens -> add_position_embeddings -> run_blocks
           -> final_norm -> project_to_logits -> next_token_loss

Because the steps are plain functions operating on a parameter holder, you can
experiment (swap the ordering, insert a probe, feed a different normalisation)
without subclassing anything or touching the model definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ModelConfig
from .loss import next_token_loss
from .model.block import transformer_block_forward
from .model.gpt import GPTModel


@dataclass
class LMOutput:
    """Result of a forward pass: vocabulary scores, and the loss if we had targets."""

    logits: torch.Tensor  # (batch, time, vocab_size)
    loss: torch.Tensor | None = None  # scalar, only when `targets` was supplied


# --------------------------------------------------------------------------- #
# Individual pipeline steps
# --------------------------------------------------------------------------- #
def embed_tokens(model: GPTModel, idx: torch.Tensor) -> torch.Tensor:
    """Step 1 - look up the learned vector for each token id."""
    return model.wte(idx)


def add_position_embeddings(model: GPTModel, token_emb: torch.Tensor) -> torch.Tensor:
    """Step 2 - tell the model *where* each token is.

    Attention on its own is order-blind: "dog bites man" and "man bites dog"
    would produce identical scores.  Adding a position-dependent vector to every
    token is what injects word order back into the model.
    """
    time = token_emb.size(1)
    positions = torch.arange(time, device=token_emb.device)
    return model.drop(token_emb + model.wpe(positions))


def run_blocks(model: GPTModel, x: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    """Step 3 - push the activations through every transformer block in turn."""
    for block in model.blocks:
        x = transformer_block_forward(x, block, cfg)
    return x


def final_norm(model: GPTModel, x: torch.Tensor) -> torch.Tensor:
    """Step 4 - normalise before the head (pre-norm architecture needs this)."""
    return model.ln_f(x)


def project_to_logits(model: GPTModel, x: torch.Tensor) -> torch.Tensor:
    """Step 5 - map each hidden vector to one score per vocabulary entry."""
    return model.lm_head(x)


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def forward_pass(
    model: GPTModel,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
    cfg: ModelConfig | None = None,
) -> LMOutput:
    """Run the whole pipeline and optionally compute the training loss.

    Parameters
    ----------
    model
        Parameter holder built from ``cfg``.
    idx
        Token ids of shape ``(batch, time)``.
    targets
        Next-token ids of shape ``(batch, time)``.  When given, the loss is
        computed as the last step.
    """
    cfg = cfg or model.cfg

    x = embed_tokens(model, idx)
    x = add_position_embeddings(model, x)
    x = run_blocks(model, x, cfg)
    x = final_norm(model, x)
    logits = project_to_logits(model, x)

    loss = next_token_loss(logits, targets) if targets is not None else None
    return LMOutput(logits=logits, loss=loss)
