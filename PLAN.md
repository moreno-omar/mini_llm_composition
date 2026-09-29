# PLAN.md — Phase 1 implementation guide

Companion to `ROADMAP.md` (Phase: *Creating working skeleton of app*) and
`TASKS.md`. This file explains **how** Phase 1 is built and **why** each piece
exists in the shape it does.

## Guiding principle: composition, not inheritance

`SPEC.md` forbids the classic pattern where one class owns a giant `forward()`
that does everything. The rule used throughout this code base:

| Layer | Responsibility | Example |
|---|---|---|
| `config.py` | plain data only | `ModelConfig`, `TrainConfig` |
| `model/*.py` | **parameter holders** (`nn.Module`) | `CausalSelfAttention`, `MLP`, `TransformerBlock`, `GPTModel` |
| `pipeline.py` | the actual maths, as small free functions | `embed_tokens`, `run_blocks`, `forward_pass` |
| `train.py` | thin orchestrator: build objects, loop, save | `train()` |
| `generate.py` | sampling built on the same pipeline | `generate_ids` |
| `main.py` | flag parsing only | `cmd_train` |

Consequence: `GPTModel.forward` is a **one-line delegate** to
`pipeline.forward_pass`. Nothing is subclassed anywhere in the project.

## Step order and rationale

1. **`config.py`** — hyper-parameters as serialisable data. A checkpoint stores
   its own config, so any run can be reproduced exactly.
2. **`tokenizer.py`** — wraps GPT-2's byte-level BPE from HuggingFace.
   `SPEC.md` lists "write a tokenizer" as a non-goal, so this is intentionally a
   thin adapter (≈70 lines).
3. **`data.py`** — WikiText-103 → one flat stream of token ids, cached to
   `uint16` binary files, read back through `numpy.memmap`. Targets are just the
   stream shifted one position: that *is* next-token prediction.
4. **`model/attention.py`** — multi-head causal attention. Masked so token *t*
   cannot see the future.
5. **`model/mlp.py`** — position-wise expand → GELU → project, the non-linear
   part of the block.
6. **`model/block.py`** — pre-norm residual block = the two sub-layers above.
7. **`model/gpt.py`** — the model as a statement of which tensors exist
   (`wte`, `wpe`, `blocks`, `ln_f`, `lm_head`), plus weight initialisation.
8. **`pipeline.py`** — the forward pass as a readable chain of five steps, then
   optional loss.
9. **`loss.py`** — cross-entropy with the `(B,T,V) → (B*T,V)` reshape isolated
   in one documented place.
10. **`train.py`** — AdamW with decay/no-decay groups, warm-up + cosine LR,
    gradient accumulation, clipping, periodic eval, best/last checkpoints.
11. **`generate.py`** — autoregressive sampling with temperature / top-k.
12. **`main.py`** — `prepare` / `train` / `generate` / `chat`.

## Verification strategy

Per `AGENTS.md` there are **no unit tests**. Verification is by direct
execution:

```bash
.venv/bin/python -m minillm prepare --max-docs 200
.venv/bin/python -m minillm train --max-docs 200 --max-iters 20 \
    --block-size 64 --n-layer 2 --n-head 4 --n-embd 128 --dropout 0.0 \
    --eval-interval 10 --eval-iters 2 --log-interval 5
.venv/bin/python -m minillm generate --checkpoint checkpoints/smoke/best.pt \
    --prompt "The history of the world" --max-new-tokens 40
```

Expected signals: initial loss ≈ `ln(50257) ≈ 10.82`, falling during training,
and a checkpoint that reloads and generates text.

## Phase 1 exit criteria mapping

| Roadmap exit criterion | Where it is satisfied |
|---|---|
| app runs | `python -m minillm ...` CLI, verified above |
| comments explaining *why* | docstring "Why this file exists" in every module |
| small LLM trained on WikiText | `prepare` + `train` on `Salesforce/wikitext` |
| can ask it basic questions | `generate` and `chat` sub-commands |
