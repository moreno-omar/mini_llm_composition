"""minillm - a small, composable GPT you can train from scratch.

The public surface is intentionally tiny: configuration objects, the model
parameter holder, the forward pipeline, and the four pipeline stages
(prepare / train / generate).

Example
-------
>>> from minillm import Config, GPTModel, forward_pass
>>> import torch
>>> cfg = Config()
>>> model = GPTModel(cfg.model)
>>> idx = torch.randint(0, cfg.model.vocab_size, (2, 16))
>>> out = forward_pass(model, idx)
>>> out.logits.shape
torch.Size([2, 16, 50257])
"""

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data import TokenDataset, prepare_dataset
from .generate import generate_ids, generate_text
from .loss import next_token_loss
from .model import GPTModel
from .pipeline import LMOutput, forward_pass
from .tokenizer import Tokenizer
from .train import train

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "GPTModel",
    "LMOutput",
    "TokenDataset",
    "Tokenizer",
    "forward_pass",
    "generate_ids",
    "generate_text",
    "next_token_loss",
    "prepare_dataset",
    "train",
]

__version__ = "0.1.0"
