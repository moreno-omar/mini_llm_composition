"""Thin wrapper around a pre-trained HuggingFace tokenizer.

Why this file exists
--------------------
``SPEC.md`` explicitly lists "create a tokenizer" as a *non-goal*: writing a
correct BPE implementation is a distraction from the goal of studying the
transformer itself.  So we take an already well established tokenizer (GPT-2's
byte-level BPE) and wrap it in a tiny class with the three operations the rest
of the pipeline needs: ``encode``, ``decode`` and ``vocab_size``.

The wrapper also isolates the rest of the code from the underlying library, so
swapping in a different tokenizer later means touching only this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer as HFTokenizer

# GPT-2 uses this special token to separate documents in a concatenated corpus.
END_OF_TEXT = "<|endoftext|>"


class Tokenizer:
    """Minimal text <-> token-id interface backed by ``tokenizers``."""

    def __init__(self, name: str = "gpt2") -> None:
        self.name = name
        # ``from_pretrained`` downloads the `tokenizer.json` from the Hub and
        # caches it locally, so this is a one-time network access.
        self._tokenizer = HFTokenizer.from_pretrained(name)

        eos_id = self._tokenizer.token_to_id(END_OF_TEXT)
        if eos_id is None:
            raise ValueError(f"{name!r} has no {END_OF_TEXT!r} token; cannot separate documents")
        self.eos_id: int = eos_id

    # -- introspection ----------------------------------------------------- #
    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens, including special tokens."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    # -- conversion -------------------------------------------------------- #
    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token ids."""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        """Convert token ids back into text (special tokens skipped)."""
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many strings at once using the fast (Rust) implementation."""
        encodings = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def save(self, path: str | Path) -> None:
        """Persist the tokenizer so data preparation is reproducible offline."""
        self._tokenizer.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path, name: str = "local") -> "Tokenizer":
        """Load a previously saved tokenizer."""
        obj = cls.__new__(cls)
        obj.name = name
        obj._tokenizer = HFTokenizer.from_file(str(path))
        obj.eos_id = obj._tokenizer.token_to_id(END_OF_TEXT)
        return obj
