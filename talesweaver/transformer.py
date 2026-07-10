import jax.numpy as jnp
import flax.nnx as nnx


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ff1 = nnx.Linear(embed_dim, ff_dim, rngs=rngs)
        self.ff2 = nnx.Linear(ff_dim, embed_dim, rngs=rngs)

    def __call__(self, x, mask=None):
        # pre-LN transformer block: attention + FFN, each with a residual
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + attn_out

        ff_out = self.ff2(nnx.gelu(self.ff1(self.ln2(x))))
        x = x + ff_out
        return x
