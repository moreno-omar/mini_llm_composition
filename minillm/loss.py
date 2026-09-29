"""Next-token prediction loss.

Why this file exists
--------------------
The training objective of a GPT model is one line of cross-entropy, but it is
easy to get the reshaping wrong.  Isolating it here keeps that subtle detail in
one documented place: the model produces ``(batch, time, vocab)`` logits while
``cross_entropy`` expects ``(samples, classes)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Positions we do not want to score (e.g. padding) can be marked with this id.
IGNORE_INDEX = -100


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy between predicted logits and the true next token.

    Parameters
    ----------
    logits
        Unnormalised scores of shape ``(batch, time, vocab)``.
    targets
        True next-token ids of shape ``(batch, time)``.

    Returns
    -------
    A scalar tensor (cross-entropy is already averaged over all positions).
    """
    if logits.dim() != 3:
        raise ValueError(f"expected logits of shape (B, T, V), got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"targets {tuple(targets.shape)} must match logits[:2] {tuple(logits.shape[:2])}"
        )

    batch, time, vocab = logits.shape
    # Flatten batch and time into a single "samples" axis; the vocabulary stays
    # as the class axis.
    return F.cross_entropy(
        logits.reshape(batch * time, vocab),
        targets.reshape(batch * time),
        ignore_index=ignore_index,
    )
