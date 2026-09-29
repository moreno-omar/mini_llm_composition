"""Command-line interface: prepare data, train, generate, chat.

Why this file exists
--------------------
``SPEC.md`` wants a *functional* application, not a pile of library code.  This
module is the single entry point that wires the pipeline stages into the four
commands a user actually runs:

    python -m minillm prepare    # download + tokenise WikiText (once)
    python -m minillm train      # train the model
    python -m minillm generate   # sample text from a checkpoint
    python -m minillm chat       # interactive, keeps the model in memory

Each sub-command only translates flags into a :class:`~minillm.config.Config`
and calls one function.  No logic lives here.
"""

from __future__ import annotations

import argparse

from .config import Config
from .data import prepare_dataset
from .generate import generate_text
from .tokenizer import Tokenizer
from .train import load_checkpoint, train


# --------------------------------------------------------------------------- #
# Shared arguments
# --------------------------------------------------------------------------- #
def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="only tokenise the first N documents per split (fast smoke test)",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")


def _add_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def _config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from whichever flags the chosen sub-command defined."""
    cfg = Config()

    if hasattr(args, "data_root"):
        cfg.data.root = args.data_root
        cfg.data.dataset_name = args.dataset_name
        cfg.data.dataset_config = args.dataset_config
        cfg.data.tokenizer_name = args.tokenizer
        cfg.data.max_docs = args.max_docs

    for flag, target in (
        ("block_size", "block_size"),
        ("n_layer", "n_layer"),
        ("n_head", "n_head"),
        ("n_embd", "n_embd"),
        ("dropout", "dropout"),
    ):
        if hasattr(args, flag):
            setattr(cfg.model, target, getattr(args, flag))

    for flag in (
        "run_name",
        "out_dir",
        "batch_size",
        "grad_accum_steps",
        "learning_rate",
        "max_iters",
        "warmup_iters",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "grad_clip",
        "seed",
        "device",
        "dtype",
    ):
        if hasattr(args, flag):
            setattr(cfg.train, flag, getattr(args, flag))

    return cfg


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_prepare(args: argparse.Namespace) -> None:
    """Task 1: download WikiText and cache it as flat token streams."""
    cfg = _config_from_args(args)
    tokenizer = Tokenizer(cfg.data.tokenizer_name)
    cfg.model.vocab_size = tokenizer.vocab_size
    prepare_dataset(cfg.data, tokenizer, force=args.force, max_docs=args.max_docs)
    print(f"[prepare] ready. vocabulary size = {tokenizer.vocab_size:,}")


def cmd_train(args: argparse.Namespace) -> None:
    """Task: train the model (the loop itself lives in ``train.py``)."""
    cfg = _config_from_args(args)
    tokenizer = Tokenizer(cfg.data.tokenizer_name)
    cfg.model.vocab_size = tokenizer.vocab_size

    # Convenience: tokenise on demand so `train` works on a fresh checkout.
    if not (cfg.data.train_bin.exists() and cfg.data.val_bin.exists()):
        print("[train] token cache missing, preparing the dataset first ...")
        prepare_dataset(cfg.data, tokenizer, max_docs=cfg.data.max_docs)

    train(cfg)


def cmd_generate(args: argparse.Namespace) -> None:
    """Sample a completion from a trained checkpoint."""
    device = args.device
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer(cfg.data.tokenizer_name)

    print(f"[generate] checkpoint={args.checkpoint} device={device}")
    print("-" * 60)
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            cfg=cfg.model,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )
    print("-" * 60)


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive loop - loads the model once and keeps it resident."""
    device = args.device
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer(cfg.data.tokenizer_name)
    model.eval()

    print(f"[chat] loaded {args.checkpoint} on {device}. Ctrl-D or 'exit' to quit.")
    while True:
        try:
            prompt = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye")
            return
        if prompt.strip().lower() in {"exit", "quit"}:
            print("[chat] bye")
            return

        reply = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            cfg=cfg.model,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\nmodel> {reply}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minillm",
        description="A small, compositional GPT you can train on WikiText-103.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare ------------------------------------------------------------- #
    p_prepare = sub.add_parser("prepare", help="download + tokenise the dataset")
    _add_data_args(p_prepare)
    p_prepare.add_argument("--force", action="store_true", help="re-tokenise from scratch")
    p_prepare.set_defaults(func=cmd_prepare)

    # train --------------------------------------------------------------- #
    p_train = sub.add_parser("train", help="train the model")
    _add_data_args(p_train)
    _add_model_args(p_train)
    _add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # generate ------------------------------------------------------------ #
    p_generate = sub.add_parser("generate", help="sample text from a checkpoint")
    p_generate.add_argument("--prompt", default="Once upon a time")
    p_generate.add_argument("--device", default="auto")
    _add_sample_args(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    # chat ---------------------------------------------------------------- #
    p_chat = sub.add_parser("chat", help="interactive generation")
    p_chat.add_argument("--device", default="auto")
    _add_sample_args(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # "auto" is resolved here so every command shares one code path.
    if hasattr(args, "device") and args.device == "auto":
        args.device = Config().train.resolve_device()

    args.func(args)


if __name__ == "__main__":
    main()
