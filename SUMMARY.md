# SUMMARY.md

**Last updated:** 2026-09-29 13:35

## What this project is

A small GPT-style language model built **from scratch with PyTorch**, trained on
WikiText-103, written so that the *composition* of small pieces is visible
instead of hidden behind inheritance. The goal is study and experimentation,
not performance.

## How it works — the short version

A language model is asked one question over and over: *given the tokens so far,
what comes next?* Everything else is machinery around that.

1. **Text → numbers.** GPT-2's BPE tokenizer (`tokenizer.py`) cuts text into
   ~50,000 possible sub-word tokens. We do not write this ourselves (`SPEC.md`
   non-goal).
2. **Numbers → one long stream.** `data.py` downloads WikiText-103, tokenises it
   once, and caches the ids as binary files. Training then samples random
   windows. The "answer" for a window is the same window shifted by one token.
3. **Numbers → vectors.** `wte` gives every token a learned vector; `wpe` adds a
   second vector saying *where* it is, because attention alone is order-blind.
4. **Mixing.** Each of the `n_layer` blocks does two things:
   - **Attention** (`model/attention.py`): every token looks at all *earlier*
     tokens and pulls in the ones that seem relevant. A triangular mask makes
     looking ahead impossible — otherwise the model would read the answer.
   - **Feed-forward** (`model/mlp.py`): each token, on its own, is expanded 4x,
     pushed through GELU, and projected back. This is where most parameters and
     most "thinking" capacity live.
   Both are wrapped with a residual add (`x = x + f(x)`) so gradients reach the
   early layers.
5. **Vectors → scores.** A final LayerNorm plus `lm_head` produces one score per
   vocabulary entry. Softmax turns scores into probabilities.
6. **Error → gradient.** `loss.py` computes cross-entropy against the true next
   token; backprop adjusts every weight to make the true token more likely.
7. **Sampling.** `generate.py` repeats steps 3–5 and draws tokens, with
   `temperature` (sharpness) and `top_k` (cut off the tail) as knobs.

## The composition idea (the point of the project)

| File | What it is |
|---|---|
| `config.py` | plain data: model shape, data paths, optimiser knobs |
| `model/attention.py` | `CausalSelfAttention` = **parameters only**; `causal_self_attention()` = the maths |
| `model/mlp.py` | `MLP` = parameters; `feed_forward()` = the maths |
| `model/block.py` | `TransformerBlock` = parameters; `transformer_block_forward()` = the maths |
| `model/gpt.py` | `GPTModel` = which parameters exist, nothing more |
| `pipeline.py` | the forward pass as 5 named steps + optional loss |
| `train.py` | thin orchestrator: build, loop, checkpoint |
| `generate.py` | sampling on top of the same pipeline |
| `main.py` | CLI flags, no logic |

Because the maths are free functions, you can reorder or replace a step — for
example swap the LayerNorm placement or probe intermediate activations — without
touching a class hierarchy.

## Key steps taken, and why

| # | Step | Why it matters |
|---|---|---|
| 1 | Config objects as data | A checkpoint stores its own config, so runs are reproducible |
| 2 | Pre-built GPT-2 tokenizer | Tokenizer is a non-goal; correctness comes free |
| 3 | Tokenise once → binary cache | Tokenising 119.7M tokens takes minutes; training should not |
| 4 | `memmap` the cache | The corpus never needs to fit in RAM |
| 5 | Separate parameters from maths | Makes the forward pass readable and editable |
| 6 | Pre-norm + residuals | Standard GPT-2 choice; stable enough to train without tricks |
| 7 | Weight tying (`lm_head.weight = wte.weight`) | Saves ~19M params and regularises |
| 8 | Damped init on residual projections | Keeps activations from growing with depth |
| 9 | AdamW with decay/no-decay groups | Decaying LayerNorm/biases hurts training |
| 10 | Warm-up + cosine LR | Warm-up prevents a destructive first step; cosine settles late |
| 11 | Gradient accumulation | Reaches a large effective batch on one GPU |
| 12 | Grad clipping | One bad batch cannot wreck the weights |
| 13 | Device/dtype auto-detection | Same code path on CPU, CUDA and Apple Silicon |

## Verified results

```text
[data] wrote 119,721,490 tokens -> data/tokenized/wikitext-103-raw-v1_gpt2/train.bin
[data] wrote     251,049 tokens -> data/tokenized/wikitext-103-raw-v1_gpt2/val.bin
[train] device=cuda dtype=torch.bfloat16 | tokens: train=11,996 val=13,567 | params: 6.84M
[train] step 5/20  loss 10.7354   <- ln(50257) = 10.82, so learning started
[eval ] step 20    train 9.7343  val 10.0024
[generate] The history of the world by of armed, 1923 the for 4 the Chronicles ...
```

The generated text is nonsense — expected from 20 steps on 12k tokens. It proves
the *mechanics* work: load → forward → softmax → sample → decode.

## Commands

```bash
.venv/bin/python -m minillm prepare                     # once
.venv/bin/python -m minillm train --run-name wikitext   # train
.venv/bin/python -m minillm generate --checkpoint checkpoints/wikitext/best.pt --prompt "The "
.venv/bin/python -m minillm chat   --checkpoint checkpoints/wikitext/best.pt
```

## Suggestions (to improve)

1. **Use `F.scaled_dot_product_attention`** in `causal_self_attention`. It is a
   drop-in replacement for steps 3–6 and uses Flash-Attention kernels. Keep the
   manual version as the "learning" path behind a config flag.
2. **Training time estimate first.** 119.7M tokens at `block_size 256` is ~468k
   tokens per step with the default batch; 5000 iterations is a starting point,
   not a target. Log an ETA at step 1.
3. **Curriculum on `roneneldan/TinyStories`.** The `DataConfig` already
   parameterises the dataset; TinyStories teaches coherent sentences in minutes
   on a small model, which is far more satisfying than early WikiText output.
4. **Resume support is half-wired.** `train()` accepts `resume_from`, but no CLI
   flag exposes it. Add `--resume`.
5. **Add `torch.compile(model)`** behind a flag for a free 1.3–2x speed-up on
   Linux+CUDA.
6. **Gradient checkpointing** (`torch.utils.checkpoint`) to trade compute for
   memory and allow a bigger `block_size`.
7. **A small `eval` command** that reports perplexity (`exp(loss)`) on the
   validation split — loss in nats is hard to read intuitively.
8. **`rich` for the CLI** (already listed in `SPEC.md` future additions) to show
   a live progress bar and loss sparkline.
9. **Config file loading.** `Config.load()` exists but no command uses it; add
   `--config path.json` so experiments can be checked in as files.
10. **BPE vocabulary trim.** 50257 GPT-2 tokens include many rare ones; a smaller
    vocabulary speeds up the softmax (which dominates cost at this scale).

## Possible problems

1. **`git` will try to track `data/` and `checkpoints/`.** The token cache is
   229 MB. Add both to `.gitignore` (done) and never commit them.
2. **`uint16` is hard-coded when reading** (`TokenDataset._load`). It is correct
   for GPT-2's 50,257 tokens, but a tokenizer with >65,535 tokens would silently
   produce garbage. `_dtype_for_vocab` picks the write dtype but the reader does
   not infer it — the filename should carry the dtype.
3. **`datasets` reads the whole text column into a Python list**, which is heavy
   for WikiText-103. Streaming or sharded tokenisation would cap peak RAM.
4. **CPU training is impractical.** ~0.03 s/step on CUDA for the smoke model
   becomes seconds per step on CPU for the full config; `max_iters` defaults will
   look like a hang.
5. **`bfloat16` autocast is CUDA-only in this code** (`use_amp = device == "cuda"`).
   CPU and MPS silently run float32 — correct, but slower than expected.
6. **No `block_size` validation against the tokenizer.** Config errors surface
   only when a batch is requested.
7. **`DataConfig.max_docs` changes the cache directory**, so capping documents
   creates a second copy of the corpus. Fine for smoke tests, but disk adds up.
8. **Checkpoint incompatibility.** `tie_embeddings` changes the state dict keys
   present; loading a tied checkpoint into an untied config will fail. The config
   travels with the checkpoint, so use `load_checkpoint` rather than building a
   model by hand.
9. **WikiText-103 is Wikipedia prose**, so a "chat" prompt will never produce a
   conversation — it continues encyclopedia text. Expectation management, and a
   reason to add instruction tuning or TinyStories later.
