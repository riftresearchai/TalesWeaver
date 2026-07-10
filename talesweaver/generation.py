import os

import jax
import jax.numpy as jnp
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
_END_TOKEN = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]


def generate_text(model, start_tokens, max_new_tokens=50, temperature=1.0, top_k=40, seed=None):
    tokens = list(start_tokens)

    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')
    key = jax.random.PRNGKey(seed)

    repeat_streak = 0
    prev_token = None

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # right-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / max(temperature, 1e-5)

        # discourage runaway repetition of the same token
        if prev_token is not None and repeat_streak >= 2:
            next_token_logits = next_token_logits.at[prev_token].set(-jnp.inf)

        if top_k is not None:
            top_logits, top_indices = jax.lax.top_k(next_token_logits, top_k)
            key, subkey = jax.random.split(key)
            choice = jax.random.categorical(subkey, top_logits)
            next_token = int(top_indices[choice])
        else:
            key, subkey = jax.random.split(key)
            next_token = int(jax.random.categorical(subkey, next_token_logits))

        if next_token == _END_TOKEN:
            break

        repeat_streak = repeat_streak + 1 if next_token == prev_token else 0
        prev_token = next_token

        tokens.append(next_token)

    return tokenizer.decode(tokens)


def generate_story(model, story_prompt, temperature=1.0, max_new_tokens=100):
    start_tokens = tokenizer.encode(story_prompt)[:model.maxlen]
    return generate_text(model, start_tokens, max_new_tokens=max_new_tokens, temperature=temperature)
