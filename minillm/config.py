"""Central configuration objects for the whole pipeline.

Why this file exists
--------------------
Every tunable value (model shape, data location, optimiser settings) lives in a
small dataclass instead of being scattered across the code base.  A single
:class:`Config` can be serialised to JSON next to a checkpoint, which makes a
training run reproducible: the same config that *trained* a model is the config
that *loads* it again.

Keeping hyper-parameters in plain data (not in the classes that use them) also
supports the project goal of "composition over inheritance": the model modules
receive a config, they do not own global state.
"""

from __future__ import annotations

import json
import torch
from dataclasses import asdict, dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Shape and regularisation of the transformer itself.

    The defaults describe a small GPT-2 style model (~10M parameters) that can
    be trained in a reasonable time on a single GPU, which matches the
    "functional, clarity over optimisation" goal of ``SPEC.md``.
    """

    vocab_size: int = 50257  # GPT-2 BPE vocabulary (see ``tokenizer.py``)
    block_size: int = 256  # maximum context length the model can attend to
    n_layer: int = 6  # number of stacked transformer blocks
    n_head: int = 6  # attention heads per block
    n_embd: int = 384  # residual stream width
    dropout: float = 0.1
    bias: bool = True  # bias terms in Linear/LayerNorm layers
    tie_embeddings: bool = True  # share wte and lm_head weights (fewer params)

    def __post_init__(self) -> None:
        # A silently mis-split attention would only show up as garbage loss,
        # so fail loudly at construction time instead.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        """Width of a single attention head."""
        return self.n_embd // self.n_head


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the text comes from and how it is turned into token ids."""

    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"
    text_column: str = "text"
    root: str = "data"  # everything cached below here
    max_docs: int | None = None  # cap documents per split (useful for smoke runs)

    @property
    def tokenized_dir(self) -> Path:
        """Directory that holds the ``.bin`` token files for this dataset.

        A capped run (``max_docs``) gets its own folder, otherwise a quick smoke
        test would silently leave a truncated corpus behind that a later full
        training run would happily reuse.
        """
        suffix = f"_{self.dataset_config}_{self.tokenizer_name}"
        if self.max_docs is not None:
            suffix += f"_maxdocs{self.max_docs}"
        return Path(self.root) / "tokenized" / suffix.lstrip("_")

    @property
    def train_bin(self) -> Path:
        return self.tokenized_dir / "train.bin"

    @property
    def val_bin(self) -> Path:
        return self.tokenized_dir / "val.bin"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation and bookkeeping knobs for the training loop."""

    batch_size: int = 12
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100  # batches averaged when estimating loss
    log_interval: int = 25
    seed: int = 1337
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "auto"  # "auto" | "float32" | "bfloat16" | "float16"
    out_dir: str = "checkpoints"
    run_name: str = "run"

    def resolve_device(self) -> str:
        """Turn ``"auto"`` into a concrete device string."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Bundle of every sub-config, so one object can drive the whole pipeline."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the config as pretty JSON (stored next to checkpoints)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
