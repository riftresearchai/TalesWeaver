# Model Card: TalesWeaver

## Overview

TalesWeaver is a small decoder-only transformer trained from scratch for
short story generation.

## Architecture

Pre-LN decoder-only transformer with learned token + position embeddings,
GPT-2 BPE tokenizer (`tiktoken`, vocab size 50,257).

| | |
|---|---|
| Layers | 6 |
| Embedding dim | 192 |
| Attention heads | 6 |
| Feed-forward dim | 512 |
| Context length | 128 tokens |
| Parameters | ~21.4M |

Each block: `LayerNorm -> self-attention -> residual -> LayerNorm -> FFN (GELU) -> residual`,
followed by a final LayerNorm before the (untied) output projection.

See `talesweaver/transformer.py` and `talesweaver/model.py`.

## Training data

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(GPT-4-generated validation split), a synthetic dataset of short stories
using a restricted vocabulary aimed at the level a 3-4 year old would
understand.

## Training procedure

- 20,000 stories, 3 epochs (1,875 steps), batch size 32, sequence length 128
- AdamW, warmup + cosine decay schedule, peak LR 3e-4
- Training loss: 10.87 -> 2.82

Training code (data loading, the training loop) lives in the private
research repo, not here — this repo ships the architecture and inference
code needed to load a checkpoint and generate text, not to reproduce
training from scratch.

## Intended use

Toy story generation from a short prompt, e.g. "Once upon a time". See
`examples/generate.ipynb`.

## Limitations

- Trained on a small slice of a deliberately simple, narrow-vocabulary
  dataset.
- 128-token context window; longer prompts are truncated.
- No safety filtering or alignment; treat generations as unmoderated
  model output.
- No formal evaluation suite yet (see `evaluation/`).
