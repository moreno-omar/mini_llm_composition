"""Training loop - a thin orchestrator over the pieces built elsewhere.

Why this file exists
--------------------
This module deliberately contains **no model maths**.  It only glues together
objects that already exist:

    data batches  +  model  +  loss (inside the pipeline)  +  optimiser
        +  learning-rate schedule  +  periodic evaluation  +  checkpoints

Keeping it "thin" means the interesting parts of the project (the pipeline and
the model components) stay readable, and the loop stays boring - which is
exactly what you want from a training loop.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .config import Config, TrainConfig
from .data import TokenDataset
from .model.gpt import GPTModel
from .pipeline import forward_pass


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG we rely on, so two runs with one config match."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Pick a floating-point precision that the chosen device supports."""
    if name == "float32":
        return torch.float32
    if device != "cuda":
        return torch.float32  # CPU bfloat16 is slow; keep it simple
    if name == "auto":
        # bfloat16 has enough range for training and needs no loss scaling.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


# --------------------------------------------------------------------------- #
# Optimiser and schedule
# --------------------------------------------------------------------------- #
def configure_optimizer(model: GPTModel, train_cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to matrices.

    Biases, LayerNorm scales and embeddings are not decayed: shrinking them
    towards zero has no regularising benefit and measurably hurts training.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )


def lr_at_step(step: int, train_cfg: TrainConfig) -> float:
    """Linear warm-up followed by cosine decay down to ``min_lr``.

    Warm-up avoids a destructive first update when the loss surface is steep;
    cosine decay lets the model settle into a minimum near the end of training.
    """
    if step < train_cfg.warmup_iters:
        return train_cfg.learning_rate * (step + 1) / max(1, train_cfg.warmup_iters)

    decay_span = max(1, train_cfg.max_iters - train_cfg.warmup_iters)
    progress = min(1.0, (step - train_cfg.warmup_iters) / decay_span)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    dataset: TokenDataset,
    train_cfg: TrainConfig,
    device: str,
    generator: torch.Generator,
) -> dict[str, float]:
    """Average the loss over a few batches of each split.

    A single batch is far too noisy to read a trend from, so we average. Dropout
    is disabled here (`eval` mode) to get a deterministic, comparable number.
    """
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(train_cfg.eval_iters)
        for i in range(train_cfg.eval_iters):
            x, y = dataset.get_batch(split, train_cfg.batch_size, device, generator)
            output = forward_pass(model, x, targets=y)
            losses[i] = output.loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    cfg: Config,
    step: int,
    best_val: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Persist parameters *and* the config, so a model can always be rebuilt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "best_val_loss": best_val,
            # The optimiser state is only needed to *resume* training; dropping
            # it keeps the "best model" file small.
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str) -> tuple[GPTModel, Config, dict]:
    """Rebuild a model from a checkpoint (config included) and load its weights."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(payload["config"])
    model = GPTModel(cfg.model).to(device)
    model.load_state_dict(payload["model"])
    return model, cfg, payload


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def train(cfg: Config, resume_from: str | Path | None = None) -> Path:
    """Run the whole training procedure and return the best checkpoint path.

    This is the "thin orchestrator": read a config, build objects, loop, save.
    """
    device = cfg.train.resolve_device()
    dtype = resolve_dtype(cfg.train.dtype, device)
    set_seed(cfg.train.seed)

    out_dir = Path(cfg.train.out_dir) / cfg.train.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    print(f"[train] device={device} dtype={dtype} run={cfg.train.run_name}")

    # -- objects, all built from the same config -------------------------- #
    dataset = TokenDataset(cfg.data, cfg.model.block_size)
    model = GPTModel(cfg.model).to(device)
    optimizer = configure_optimizer(model, cfg.train)
    generator = torch.Generator().manual_seed(cfg.train.seed)

    print(
        f"[train] tokens: train={len(dataset.train):,} val={len(dataset.val):,} "
        f"| params: {model.num_parameters() / 1e6:.2f}M"
    )

    start_step, best_val = 0, float("inf")
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        start_step = payload.get("step", 0)
        best_val = payload.get("best_val_loss", float("inf"))
        print(f"[train] resumed from {resume_from} at step {start_step}")

    # float16 needs loss scaling to avoid underflowing small gradients;
    # bfloat16 and float32 do not.
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    use_amp = device == "cuda"

    best_path = out_dir / "best.pt"
    model.train()
    running_loss, running_iters = 0.0, 0
    started = time.time()

    for step in range(start_step, cfg.train.max_iters):
        apply_lr(optimizer, lr_at_step(step, cfg.train))

        # -- gradient accumulation: several small batches == one big batch -- #
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.train.grad_accum_steps):
            x, y = dataset.get_batch("train", cfg.train.batch_size, device, generator)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                output = forward_pass(model, x, targets=y)
                # Divide so the accumulated gradient equals the average over
                # the effective batch.
                loss = output.loss / cfg.train.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item()
            running_iters += 1

        # Clipping stops a single bad batch from wrecking the weights.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # -- logging ------------------------------------------------------ #
        if (step + 1) % cfg.train.log_interval == 0:
            avg = running_loss / max(1, running_iters)
            elapsed = time.time() - started
            print(
                f"[train] step {step + 1:>6}/{cfg.train.max_iters} "
                f"loss {avg:.4f} lr {lr_at_step(step, cfg.train):.2e} "
                f"| {elapsed / (step + 1 - start_step):.2f}s/step"
            )
            running_loss, running_iters = 0.0, 0

        # -- periodic evaluation + checkpointing --------------------------- #
        if (step + 1) % cfg.train.eval_interval == 0 or (step + 1) == cfg.train.max_iters:
            losses = estimate_loss(model, dataset, cfg.train, device, generator)
            print(f"[eval ] step {step + 1:>6} train {losses['train']:.4f} val {losses['val']:.4f}")
            save_checkpoint(out_dir / "last.pt", model, cfg, step + 1, best_val)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(best_path, model, cfg, step + 1, best_val)

    print(f"[train] done. best val loss {best_val:.4f} -> {best_path}")
    return best_path
