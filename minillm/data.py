"""Turn a HuggingFace text dataset into flat token streams and sample batches.

Why this file exists
--------------------
A language model does not consume "documents"; it consumes one very long stream
of token ids and learns to predict the next id at every position.  So the job of
this module is:

1. **Download** the raw text (WikiText-103 by default).
2. **Tokenise** it once and cache the result as a raw binary file on disk.
   Tokenising is slow, so we never want to do it on every run.
3. **Sample** random ``(input, target)`` windows from that cache.  The target is
   simply the input shifted one token to the left - that is the entire trick
   behind next-token prediction.

Using ``numpy.memmap`` means the corpus never has to fit in RAM.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig
from .tokenizer import Tokenizer

# `datasets` reads this environment variable to decide where to cache downloads.
# Setting it before the import keeps the whole corpus inside `data/`; passing
# `cache_dir=` directly has changed across major versions of the library.
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(DataConfig().root) / "hf_cache"))


# --------------------------------------------------------------------------- #
# Vocabulary -> integer dtype
# --------------------------------------------------------------------------- #
def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Smallest unsigned int that can hold every token id.

    Small dtype = smaller cache file = more of the corpus cached by the OS.
    GPT-2's 50k vocabulary fits comfortably in 16 bits.
    """
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


# --------------------------------------------------------------------------- #
# Preparation (run once, then cached)
# --------------------------------------------------------------------------- #
def _encode_split(texts: list[str], tokenizer: Tokenizer) -> np.ndarray:
    """Tokenise a list of documents and join them into one flat id array.

    Each document is followed by the end-of-text token, which teaches the model
    where documents begin and end instead of blurring them together.
    """
    # Batched calls keep the Rust tokenizer busy; a chunk is one numpy array so
    # we never hold a Python list of millions of ints.
    chunks: list[np.ndarray] = []
    batch_size = 1000
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for ids in tokenizer.encode_batch(batch):
            ids.append(tokenizer.eos_id)
            chunks.append(np.asarray(ids, dtype=np.int64))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def prepare_dataset(
    data_cfg: DataConfig,
    tokenizer: Tokenizer,
    force: bool = False,
    max_docs: int | None = None,
) -> tuple[Path, Path]:
    """Download, tokenise and cache the corpus. Returns ``(train_bin, val_bin)``.

    Parameters
    ----------
    force
        Re-tokenise even if the ``.bin`` cache already exists.
    max_docs
        Only tokenise the first ``N`` documents of each split.  Handy for a
        fast smoke test without waiting for 100M tokens.
    """
    train_path, val_path = data_cfg.train_bin, data_cfg.val_bin

    if not force and train_path.exists() and val_path.exists():
        return train_path, val_path

    # Imported lazily: `datasets` is heavy and only needed for this step.
    from datasets import load_dataset

    print(f"[data] downloading {data_cfg.dataset_name} / {data_cfg.dataset_config} ...")
    dataset = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config)

    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    data_cfg.tokenized_dir.mkdir(parents=True, exist_ok=True)

    for split, path in (("train", train_path), ("validation", val_path)):
        if split not in dataset:
            raise ValueError(f"split {split!r} not found in {data_cfg.dataset_name}")

        texts = dataset[split][data_cfg.text_column]
        if max_docs is not None:
            texts = texts[:max_docs]

        print(f"[data] tokenising {split} ({len(texts)} documents) ...")
        ids = _encode_split(texts, tokenizer)
        ids.astype(dtype).tofile(path)
        print(f"[data] wrote {len(ids):,} tokens -> {path}")

    return train_path, val_path


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
class TokenDataset:
    """Random-access view over the cached token streams.

    Holds both splits and hands out training batches without ever loading the
    whole corpus into memory.
    """

    def __init__(self, data_cfg: DataConfig, block_size: int) -> None:
        self.block_size = block_size
        self.train = self._load(data_cfg.train_bin)
        self.val = self._load(data_cfg.val_bin)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"token cache {path} missing - run the `prepare` step first"
            )
        return np.memmap(path, dtype=np.uint16, mode="r")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a ``(x, y)`` pair of shape ``(batch_size, block_size)``.

        ``y`` is ``x`` shifted by one token, so predicting ``y`` *is* next-token
        prediction.
        """
        data = self.train if split == "train" else self.val
        max_start = len(data) - self.block_size - 1
        if max_start < 1:
            raise ValueError("corpus shorter than block_size; tokenise more documents")

        # randint is done on the CPU even when training on GPU: copying a small
        # batch is cheaper than round-tripping random numbers to the GPU.
        starts = torch.randint(max_start, (batch_size,), generator=generator)

        x = np.stack([np.asarray(data[i : i + self.block_size], dtype=np.int64) for i in starts])
        y = np.stack(
            [np.asarray(data[i + 1 : i + 1 + self.block_size], dtype=np.int64) for i in starts]
        )

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        # Pinned memory only speeds up a *host -> GPU* copy, so skip it on CPU
        # (where asking for it is an error on machines without an accelerator).
        if device.startswith("cuda"):
            x_t, y_t = x_t.pin_memory(), y_t.pin_memory()

        return x_t.to(device, non_blocking=True), y_t.to(device, non_blocking=True)
