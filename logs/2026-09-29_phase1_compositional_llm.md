# 2026-09-29

## 13:35 — Phase 1: build a compositional, trainable mini LLM

### Objective

Implement Phase 1 of `ROADMAP.md` ("Creating working skeleton of app") for
`/home/redram/python/mini_llm_composition`: a study-oriented, trainable GPT-style
language model built by **composition instead of inheritance**, trained on
`Salesforce/wikitext` (`wikitext-103-raw-v1`). Concretely, complete every item in
`TASKS.md`: fetch the dataset, add a config object, wrap a pre-built tokenizer,
build model components, express the forward pass as a pipeline of functions,
implement next-token-prediction loss, add a thin training-loop orchestrator, and
wire it all into one runnable app.

Guiding constraint from `SPEC.md`: the model class must be a **parameter holder
only** — no monolithic `forward()` that "does everything".

### Solution

Created the `minillm` package, split so that each file has exactly one job:

| File | Purpose |
|---|---|
| `minillm/config.py` | `ModelConfig`, `DataConfig`, `TrainConfig`, `Config` — plain serialisable data (JSON round-trip) |
| `minillm/tokenizer.py` | thin wrapper over GPT-2 byte-level BPE from HuggingFace `tokenizers` (writing a tokenizer is an explicit `SPEC.md` non-goal) |
| `minillm/data.py` | WikiText-103 → flat token stream, cached as `uint16` `.bin`, sampled via `numpy.memmap` |
| `minillm/model/attention.py` | `CausalSelfAttention` (params) + `causal_self_attention()` (maths) |
| `minillm/model/mlp.py` | `MLP` (params) + `feed_forward()` (maths) |
| `minillm/model/block.py` | `TransformerBlock` (params) + `transformer_block_forward()` (maths) |
| `minillm/model/gpt.py` | `GPTModel` — declares which tensors exist; `forward` is a one-line delegate |
| `minillm/pipeline.py` | the forward pass as 5 named steps, then optional loss |
| `minillm/loss.py` | `next_token_loss()` cross-entropy, reshape documented in one place |
| `minillm/train.py` | thin orchestrator: AdamW (decay/no-decay groups), warm-up + cosine LR, grad accumulation, clipping, eval, checkpoints |
| `minillm/generate.py` | autoregressive sampling with temperature / top-k |
| `minillm/main.py`, `__main__.py` | CLI: `prepare` / `train` / `generate` / `chat` |

Also added `PLAN.md` (which `AGENTS.md` referenced but did not exist), rewrote
`SUMMARY.md` with step-by-step rationale, and marked all `TASKS.md` items done.

### Actions Performed

1. Inspected the environment: Python 3.14.7 in `.venv`, `torch 2.14.0+cu130`,
   `tokenizers 0.23.2`, CUDA available. No `datasets` package yet.
2. Installed the missing dependency:
   `.venv/bin/pip install datasets` → `datasets 5.0.1`.
3. Created the `minillm/` package layout and all 13 source files listed above
   (each with a "Why this file exists" docstring, type hints, and docstrings on
   public functions, per `AGENTS.md`).
4. Ran a positional-embedding/indexing smoke test of the model: constructed a
   tiny 2-layer model and exercised `forward_pass`, `next_token_loss`,
   `model(x)` delegation, `generate_ids`, and `backward()`.
5. Ran a capped data preparation to validate the tokeniser → binary cache path:
   `.venv/bin/python -m minillm prepare --max-docs 200`.
6. Ran a 20-step training job on the capped cache to validate the loop, LR
   schedule, evaluation and checkpointing.
7. Reloaded the saved checkpoint and generated text to close the loop.
8. Tokenised the **full** WikiText-103 corpus into the real cache
   (`data/tokenized/wikitext-103-raw-v1_gpt2/`).
9. Regenerated `requirements.txt` from the live environment
   (`.venv/bin/pip freeze > requirements.txt`, 62 pinned entries).
10. Added `data/` and `checkpoints/` to `.gitignore` (the token cache is 229 MB).
11. Updated `TASKS.md`, created `PLAN.md`, rewrote `SUMMARY.md`, wrote this log.

### Verification Output

Model smoke test (initial loss must be ≈ `ln(50257) = 10.82`):

```text
params: 3318592
logits: (3, 16, 50257) loss: None
loss with targets: 10.845 expected ~ 10.8249
via model(): (3, 16, 50257)
generated: (3, 9)
grad ok: True
```

Full dataset preparation:

```text
[prepare] ready. vocabulary size = 50,257
[data] wrote 119,721,490 tokens -> data/tokenized/wikitext-103-raw-v1_gpt2/train.bin
[data] wrote     251,049 tokens -> data/tokenized/wikitext-103-raw-v1_gpt2/val.bin
-rw-r--r-- 1 redram redram 229M Sep 29 13:34 train.bin
```

Note: the login node env is Capsule/Linux with no `systemctl` involvement for
this task, so the relevant proof is the process output above rather than a
service status. Training run:

```text
[train] device=cuda dtype=torch.bfloat16 run=smoke
[train] tokens: train=11,996 val=13,567 | params: 6.84M
[train] step      5/20 loss 10.7354 lr 6.00e-04 | 0.07s/step
[train] step     10/20 loss 10.3210 lr 5.11e-04 | 0.04s/step
[eval ] step     10 train 10.0932 val 10.2705
[train] step     20/20 loss 9.8225 lr 6.59e-05 | 0.03s/step
[eval ] step     20 train 9.7343 val 10.0024
[train] done. best val loss 10.0024 -> checkpoints/smoke/best.pt
```

Generation from the reloaded checkpoint (nonsense, as expected after 20 steps —
it proves load → forward → sample → decode works):

```text
[generate] checkpoint=checkpoints/smoke/best.pt device=cuda
The history of the world by of armed, 1923 the for 4 the Chronicles , also , Nam
```

Clean diagnostics: `get_errors` over `minillm/` → no errors.

### References Used

1. Andrej Karpathy, *nanoGPT* — https://github.com/karpathy/nanoGPT
2. Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
3. HuggingFace, *WikiText dataset card (`Salesforce/wikitext`)* — https://huggingface.co/datasets/Salesforce/wikitext

### Optional steps

- Add `--resume` to `train` (the `resume_from` parameter already exists but is
  not exposed on the CLI) so interrupted runs can continue.
- Print an ETA after the first logged step: 119.7M tokens at the default batch is
  ~468k tokens/step, so the default 5000 iterations is a long job.
- Point `--dataset-config tiny-stories` at `roneneldan/TinyStories` for a demo
  that produces readable sentences in minutes.
- Swap `causal_self_attention` for `F.scaled_dot_product_attention` behind a
  config flag (Flash-Attention kernels, free speed-up).
- Report perplexity (`exp(loss)`) in `estimate_loss` — nats are hard to read.
- Add `rich` progress/loss display, already listed in `SPEC.md` future additions.

### Possible problems

- `TokenDataset._load` hard-codes `uint16`. Correct for GPT-2 (50,257 tokens) but
  silent corruption for any tokenizer above 65,535 tokens; the dtype should be
  inferred from the cache filename.
- `data.py` materialises the whole `text` column as a Python list during
  preparation. Peak RAM during full tokenisation is high; sharded/streaming
  tokenisation would be safer.
- `bfloat16` autocast is gated on `device == "cuda"`; CPU and MPS silently run
  float32 — correct but slower than a user might expect.
- `datasets` and `tokenizers` pull large transitive dependencies; `requirements.txt`
  is now a full environment freeze (62 entries) rather than a minimal spec.
- CPU-only training with the default config looks like a hang. A guard or an
  explicit warning would help.
- The `chat` command invites conversational prompts, but a model trained only on
  Wikipedia prose will continue encyclopedia text, never answer questions.
- `config.model.vocab_size` is only corrected in the CLI (`cmd_prepare`/
  `cmd_train`); constructing `Config()` directly and calling `train()` would use
  the hard-coded default 50257 — fine for GPT-2, wrong for any other tokenizer.
