"""Model components, exposed as one flat namespace.

Everything here is a *parameter holder*.  The functions that use these objects
live in :mod:`minillm.pipeline`.
"""

from .attention import CausalSelfAttention, causal_self_attention
from .block import TransformerBlock, transformer_block_forward
from .gpt import GPTModel
from .mlp import MLP, feed_forward

__all__ = [
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "GPTModel",
    "causal_self_attention",
    "feed_forward",
    "transformer_block_forward",
]
