from dataclasses import dataclass


@dataclass(frozen=True)
class TalesWeaverConfig:
    maxlen: int = 128
    vocab_size: int = 50257  # gpt2 tiktoken BPE
    embed_dim: int = 192
    num_heads: int = 6
    feed_forward_dim: int = 512
    num_transformer_blocks: int = 6
