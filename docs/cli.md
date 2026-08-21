# CLI Reference (`kikku.run.cli`)

This module exposes `parse_run`, the frozen `RunSpec` type, and small helpers (`parse_key_value`, `make_run_dir`). It does not import MPI at import time; `make_run_dir` optionally uses `mpi4py` when available.

---

## `parse_run`

```python
def parse_run(
    name: str,
    syntax: str,
    methods: list[str],
    modes: list[str] | None = None,
    output: str = "results",
    extra_args: dict | None = None,
) -> RunSpec:
    ...
```

### Parameters

| Parameter | Type | Default | Role |
|-----------|------|---------|------|
| `name` | `str` | — | Human-readable model label; appears in `--help` |
| `syntax` | `str` | — | Path to the model directory containing `calibration.yaml` and `settings.yaml` (resolved with `Path.resolve()`; stored on `RunSpec` as `model_dir`) |
| `methods` | `list[str]` | — | Allowed global method names; drives `--method` `choices` and validation of bare names in `--compare` |
| `modes` | `list[str]` or `None` | `None` | Which optional capabilities to register; see below |
| `output` | `str` | `'results'` | Default `--output-dir` base |
| `extra_args` | `dict` or `None` | `None` | `{'--flag-name': {**argparse_kwargs}}` for model-specific flags; values are copied into `RunSpec.extra` |

### `modes` keywords

Including a string adds the corresponding arguments when present:

| Token | Registers |
|-------|-----------|
| `'compare'` | `--compare` |
| `'sweep'` | `--sweep`, `--sweep-grids`, `--sweep-params`, `--sweep-runs`, `--warmup` / `--no-warmup`, `--experiment-set`, `--config-id`, `--run-id` |
| `'simulate'` | `--simulate`, `--n-sim`, `--seed` |
| `'mpi'` | `--mpi` |
| `'gpu'` | `--gpu` |
| `'plots'` | `--plots`, `--skip-egm-plots`, `--csv-export` |

Core flags are always registered: `--method`, tier overrides, `--override-file`, `--method-override`, `--output-dir`, `--run-tag`, `--verbose`, `--trace`.

### What `parse_run` does (order)

1. Builds `ArgumentParser`, adds core and mode-gated arguments, plus `extra_args`.
2. Parses `argv`.
3. Loads `calibration` and `settings` sections from YAML under `model_dir`.
4. Parses CLI tier overrides with enforcement; merges override file (keys matched to calib/settings keys, unknown keys go to `config` with a warning).
5. Merges dicts: **CLI overrides win over file over base** for `calib` and `settings`; `config` is file then CLI.
6. Resolves **mode**: `compare` vs `sweep` vs `single` (mutual exclusion).
7. Parses sweep-related fields (`sweep_grids`, optional `experiment_set` YAML).
8. Calls `make_run_dir` for `output_dir`.
9. Collects `extra` from `extra_args` attribute names on `args`.
10. Returns `RunSpec(...)`.

---

## `RunSpec` fields

All fields are keyword-only on the dataclass; types reflect what `parse_run` constructs.

| Field | Type | Default / source | Set by |
|-------|------|------------------|--------|
| `name` | `str` | `parse_run` `name` | positional |
| `model_dir` | `Path` | Resolved `syntax` | positional |
| `calib` | `dict` | Merged YAML + overrides | YAML + `--calib-override` + override file |
| `settings` | `dict` | Merged | YAML + `--setting-override` + file |
| `config` | `dict` | Merged | `--config-override` + file (non-tier keys) |
| `method` | `str \| None` | `None` if omitted | `--method` |
| `method_overrides` | `dict` | From parsed specs | `--method-override` |
| `methods` | `tuple` | Allowed names | `parse_run(..., methods=[...])` |
| `mode` | `str` | `'single'`, `'compare'`, or `'sweep'` | `--compare` / `--sweep` |
| `compare_methods` | `tuple \| None` | Compare specs or `None` | `--compare` |
| `output_dir` | `Path` | New run folder | `make_run_dir(--output-dir, ...)` |
| `run_tag` | `str \| None` | Informational | `--run-tag` (see `make_run_dir` note below) |
| `verbose` | `bool` | `False` | `--verbose` |
| `trace` | `bool` | `False` | `--trace` |
| `mpi` | `bool` | `False` | `--mpi` (if mode registered) |
| `gpu` | `bool` | `False` | `--gpu` |
| `simulate` | `bool` | `False` | `--simulate` |
| `n_sim` | `int` | `10000` | `--n-sim` |
| `seed` | `int` | `42` | `--seed` |
| `plots` | `bool` | `False` | `--plots` |
| `sweep_grids` | `list \| None` | Parsed ints or `None` | `--sweep-grids` (comma-separated) |
| `sweep_params` | `list` | Raw tokens | `--sweep-params` |
| `sweep_runs` | `int` | `3` | `--sweep-runs` |
| `warmup` | `bool` | `True` | `--warmup` / `--no-warmup` |
| `experiment_set` | `dict \| None` | Loaded YAML or `None` | `--experiment-set` |
| `config_id` | `str \| None` | `None` | `--config-id` |
| `run_id` | `str \| None` | `None` | `--run-id` |
| `skip_egm_plots` | `bool` | `False` | `--skip-egm-plots` |
| `csv_export` | `bool` | `False` | `--csv-export` |
| `extra` | `dict` | Model flags | `extra_args` → `getattr(args, ...)` |

For the conceptual grouping of fields, see the class docstring on `RunSpec` in `kikku/run/cli.py`.

`model_dir` is the same path older examples and notes called `syntax_dir`.

---

## Override tiers

**Calibration** keys must come from `calibration.yaml`’s `calibration:` section (flat dict keys). **`--calib-override`** rejects any key that belongs only to `settings`.

**Settings** keys must come from `settings.yaml`’s `settings:` section. **`--setting-override`** rejects calibration-only keys.

**Config** is a catch-all: **`--config-override`** accepts any `key=value`, but warns if the key duplicates a known calib or settings key (you may intentionally shadow).

**Override file**: a flat YAML file (`beta: 0.98`) or wrapped under an `overrides:` key. Each key is classified by membership in `calib_keys` / `settings_keys`; unknown keys are stored in `config` with a warning.

Example:

```bash
python -m examples.durables.run \
  --calib-override beta=0.98 \
  --setting-override n_a=8000 \
  --config-override debug_dump=1
```

---

## Modes

- **`single`**: Default. No `--compare`, no `--sweep`.
- **`compare`**: `--compare SPEC [SPEC ...]`. Each `SPEC` is either a bare method name (must be in `methods`) or a dynx-style override such as `keeper_cons.upper_envelope=FUES`.
- **`sweep`**: `--sweep` enables sweep mode. Grid construction is the example’s job: `run.sweep_params` holds raw CLI tokens (e.g. `n_a=100,200`); legacy `--sweep-grids` fills `run.sweep_grids` with integers. See [sweep.md](sweep.md) for `param_grid` and `sweep()`.

---

## Extension: `extra_args`

Pass a dict mapping flag strings to `argparse.add_argument` keyword arguments:

```python
run = parse_run(
    name="my_model",
    syntax="syntax/separable",
    methods=["FUES"],
    extra_args={
        "--my-flag": {"action": "store_true", "help": "Toggle feature"},
        "--my-value": {"type": float, "default": 1.0},
    },
)
# run.extra["my_flag"], run.extra["my_value"]
```

The attribute name is derived from the flag: strip leading `-`, replace `-` with `_`. Only flags listed in `extra_args` appear in `extra`.

---

## Output directories: `make_run_dir`

```python
make_run_dir(base_dir, tag=None) -> str
```

Creates **`base_dir/YYYY-MM-DD/NNN/`** with `NNN` a zero-padded counter that increments until a non-existent directory is found. Returns the path as a string.

**MPI safety**: If `mpi4py` is importable and world size > 1, rank 0 creates the directory and **broadcasts** the path so all ranks share the same folder.

**`tag`**: Accepted for backward compatibility; **currently ignored** for path construction. Passing it emits `DeprecationWarning`. Use `run.run_tag` for logging or filenames in your example if needed.

---

## Helpers

| Function | Purpose |
|----------|---------|
| `parse_key_value(['a=1', 'b=true'])` | Parse `key=value` list to dict with YAML typing (ints, floats, bools) |
| `make_run_dir` | Timestamped incremental run folder (MPI-aware) |
