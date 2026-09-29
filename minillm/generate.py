"""Text generation (sampling) on top of the forward pipeline.

Why this file exists
--------------------
Training only ever *scores* tokens.  To turn a trained model into something you
can actually read, we repeatedly: run the pipeline, keep the scores for the last
position, turn them into a probability distribution, and draw one token.  The
extra knobs (temperature, top-k) exist because the raw distribution is often
either boring or unhinged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model.gpt import GPTModel
from .pipeline import forward_pass
from .tokenizer import Tokenizer


@torch.no_grad()
def generate_ids(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    cfg: ModelConfig | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Autoregressively extend ``idx`` with ``max_new_tokens`` sampled tokens.

    Parameters
    ----------
    idx
        Prompt token ids of shape ``(batch, time)``.
    temperature
        ``< 1`` sharpens the distribution (safer, more repetitive),
        ``> 1`` flattens it (more surprising, less coherent). ``0`` = greedy.
    top_k
        Restrict sampling to the ``k`` most likely tokens.  Cheap way to stop
        the model wandering into its most nonsensical vocabulary.
    """
    cfg = cfg or model.cfg
    model.eval()

    # A private generator keeps sampling reproducible without disturbing the
    # global RNG used by dropout during training.
    generator = None
    if seed is not None:
        generator = torch.Generator(device=idx.device).manual_seed(seed)

    for _ in range(max_new_tokens):
        # The model cannot see more than `block_size` tokens, so crop the
        # context instead of erroring out on long prompts.
        context = idx[:, -cfg.block_size :]

        output = forward_pass(model, context, cfg=cfg)
        next_token_scores = output.logits[:, -1, :]  # only the newest position

        if temperature <= 0:
            next_token = next_token_scores.argmax(dim=-1, keepdim=True)
        else:
            next_token_scores = next_token_scores / temperature

            if top_k is not None:
                k = min(top_k, next_token_scores.size(-1))
                threshold = torch.topk(next_token_scores, k, dim=-1).values[:, -1:]
                next_token_scores = next_token_scores.masked_fill(
                    next_token_scores < threshold, float("-inf")
                )

            probs = F.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def generate_text(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    cfg: ModelConfig | None = None,
    device: str = "cpu",
    temperature: float = 1.0,
    top_k: int | None = 50,
    seed: int | None = None,
) -> str:
    """Convenience wrapper: ``str`` in, ``str`` out."""
    ids = tokenizer.encode(prompt)
    if not ids:
        # An empty prompt still needs one token to start from.
        ids = [tokenizer.eos_id]

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = generate_ids(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        cfg=cfg,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return tokenizer.decode(out[0].tolist())
