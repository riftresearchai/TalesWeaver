# Weights

To use the model locally, place a checkpoint directory here, e.g.:

```
weights/small_checkpoint.orbax/
```

and load it with `orbax.checkpoint.PyTreeCheckpointer`, matching the pattern
in `examples/generate.ipynb`.
