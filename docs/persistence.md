# Solution Persistence

`kikku.run` offers two ways to put solutions on disk. They target different needs: **portable numeric archives** vs **full dolo+ nest objects**.

---

## Two formats

| Module | Format | Best for |
|--------|--------|----------|
| `io` | Directory with `solutions.npz` + `metadata.json` | Sharing results without pickle; numpy-only consumers; legacy pipelines |
| `nest_io` | Single `.nst` **pickle** file | Saving configured `SymbolicModel` objects, full nest round-trip inside Python |

Neither format stores graph topology. If your workflow needs the stage graph after loading, reconstruct it from the period template.

---

## `save_solution` / `load_solution` (`io`)

### When to use

- You want a **human-inspectable** layout (JSON metadata) and **compressed numpy** arrays.
- You do **not** need to preserve live Python objects from dolo+ (use `nest_io` for that).

### Layout on disk

Given `save_solution(path, nest, metadata=...)`, the directory `path` contains:

| File | Contents |
|------|----------|
| `solutions.npz` | All `ndarray` leaves from the nest, keyed by dotted paths (e.g. `0.keeper_cons.dcsn.c`) |
| `metadata.json` | `skeleton` (structure mirroring the nest with array placeholders), `user_metadata`, `array_keys` |

Scalars and strings stay in the JSON skeleton; arrays are flattened into the npz.

### Round-trip

`save_solution` returns the output `Path`. `load_solution(path)` returns **`(nest, user_metadata)`** — the dict you passed as `metadata` to `save_solution`, not the internal bookkeeping (skeleton, array keys). The nest structure and array values match what was saved, modulo JSON numeric typing.

```python
from kikku.run import save_solution, load_solution

save_solution("out/run_001", nest, metadata={"method": "FUES", "note": "baseline"})
nest2, meta = load_solution("out/run_001")
```

---

## `save_nest` / `load_nest` / `nest_info` (`nest_io`)

### When to use

- Checkpointing a **solved** or **unsolved** nest that includes dolo+ stage objects.
- You are fine with **pickle** semantics (same Python version / code layout assumptions as usual for pickle).

### What is preserved

The bundle written by `save_nest` includes:

- `periods` — list of period dicts with configured stages (`SymbolicModel` and friends)
- `inter_conn` — cross-period renaming
- `solutions` — optional list of solution dicts (if `solutions=True` and the nest has them)
- `metadata` — optional user dict (parameters, timestamps, tags)

`save_nest` returns the output `Path`. `load_nest` returns a dict with `periods`, `solutions` (or empty list), `inter_conn`, `metadata`, and `graph` set to `None`.

### Parameters

```python
save_nest(nest, "model.nst", solutions=True, metadata=None)  # default includes solutions
save_nest(nest, "template.nst", solutions=False)             # periods only
```

### `nest_info`

Loads the file (including full unpickle) and returns a **light summary**: `n_periods`, `stage_names`, `has_solutions`, `metadata`, `file_size_mb`. Use it for quick inventory scripts; for huge nests prefer a dedicated tool if you need size without full load.

### Pickle caveats

- **Security**: Only `load_nest` files you trust.
- **Portability**: Pickles are not a stable interchange format across major Python or library upgrades.
- **Reproducibility**: Pair `.nst` checkpoints with pinned environments in serious workflows.

```python
from kikku.run import save_nest, load_nest, nest_info

p = save_nest(nest, "results/run.nst", metadata={"run_tag": "A1"})
restored = load_nest(p)
print(nest_info(p))
```

---

## Choosing between them

| Question | Prefer |
|----------|--------|
| Need to open arrays in Julia/Matlab or archive without Python objects? | `io.save_solution` |
| Need exact in-process nest restoration with dolo+ models? | `nest_io.save_nest` |
| Long-term archival with minimal coupling? | `io` + explicit metadata |
| Fast iteration inside one repo? | `.nst` is often simpler |

For sweep tables and paper figures, examples usually write **markdown/LaTeX** via `metrics` and keep heavy checkpoints in one of the two formats above.
