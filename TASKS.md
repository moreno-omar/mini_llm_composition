# TASKS.md - Active Sprint Queue (Phase 1)

- [x] get `WikiText-103` dataset
      -> `minillm/data.py` + `python -m minillm prepare`
      -> cached 119,721,490 train tokens / 251,049 val tokens
- [x] create `config` that will hold useful variables, like vocab_size, block size, n_layer, etc.
      -> `minillm/config.py` (`ModelConfig`, `DataConfig`, `TrainConfig`, `Config`)
- [x] create tokenizer from prebuilt one
      -> `minillm/tokenizer.py` (GPT-2 byte-level BPE via HuggingFace `tokenizers`)
- [x] create model components
      -> `minillm/model/attention.py`, `mlp.py`, `block.py`, `gpt.py`
      -> parameters only; the maths lives in `minillm/pipeline.py`
- [x] create forward pass — a pipeline of functions
      -> `minillm/pipeline.py` (`embed_tokens` -> `add_position_embeddings`
         -> `run_blocks` -> `final_norm` -> `project_to_logits`)
- [x] create `simple next-token prediction`
      -> `minillm/loss.py` (`next_token_loss`), wired as the last pipeline step
- [x] create `Training loop — thin orchestrator`
      -> `minillm/train.py` (AdamW, warm-up + cosine LR, grad accumulation,
         clipping, periodic eval, best/last checkpoints)
- [x] put them togther in a full pipeline
      -> `minillm/main.py` CLI: `prepare` / `train` / `generate` / `chat`
      -> verified end to end on CUDA (loss 10.74 -> 9.73 over 20 smoke steps)