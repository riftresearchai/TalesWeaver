import jax.numpy as jnp
import flax.nnx as nnx

from .config import TalesWeaverConfig
from .transformer import TokenAndPositionEmbedding, TransformerBlock


class TalesWeaver(nnx.Module):
    def __init__(self, config: TalesWeaverConfig = TalesWeaverConfig(), *, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.maxlen = config.maxlen

        self.embeddings = TokenAndPositionEmbedding(
            config.maxlen, config.vocab_size, config.embed_dim, rngs=rngs
        )

        self.transformer_blocks = nnx.List([
            TransformerBlock(config.embed_dim, config.num_heads, config.feed_forward_dim, rngs=rngs)
            for _ in range(config.num_transformer_blocks)
        ])

        self.final_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.output_layer = nnx.Linear(config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)

        x = self.embeddings(token_ids)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        logits = self.output_layer(x)

        return logits
