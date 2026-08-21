# The `run` Package

The `kikku.run` package sits between your model’s `solve()` and anything you write to disk. It does not implement Bellman operators, grids, or stage logic. Instead, it gives you a single frozen configuration object from the CLI, optional sweeps and MPI, tables for papers, two persistence formats, a spec-driven moment engine, and a small SMM estimation stack. Your example’s `main()` stays thin: parse once, branch on mode, call your solver.

---

## Package map

| Module | Role | Documentation |
|--------|------|----------------|
| `cli` | `parse_run` → frozen `RunSpec` | [CLI and RunSpec](cli.md) |
| `sweep` | Cartesian grids, best-of-$n$, timing, MPI | [Parameter sweeps](sweep.md) |
| `metrics` | Markdown / LaTeX tables from sweep rows | [same file as sweep examples](sweep.md) |
| `mpi` | Scatter/gather, `mpi_map`, rank checks | Used internally; see [sweep.md § MPI](sweep.md#mpi-distribution) and [`make_run_dir`](cli.md#output-directories-make_run_dir) |
| `io` | NumPy + JSON solution archives | [Persistence](persistence.md) |
| `nest_io` | Pickled `.nst` nests (dolo+ objects) | [Persistence](persistence.md) |
| `moments` | Panels or microdata → `dict[str, float]` | `moments` module docstring; [estimation guide](estimation_guide.md) |
| `estimate` | Criterion composition, cross-entropy SMM | [estimation guide](estimation_guide.md), [CE method](cross_entropy_method.md) |

---

## RunSpec: the single configuration object

`parse_run` is the only place that touches `argparse`, merges YAML with overrides, enforces calibration vs settings tiers, and creates the run directory. The `model_dir` field points to the resolved path of the directory containing `calibration.yaml` and `settings.yaml`. It returns a **frozen** `RunSpec`. Example code should treat `RunSpec` as read-only configuration: pass `run` into `solve` wrappers, sweep drivers, and plotting routines instead of threading dozens of loose variables.

Design payoff: adding a new CLI flag means editing `cli.py` and possibly extending `RunSpec`; the example file only reads `run.some_field`.

### Parameter tiers

Three dicts on `RunSpec` mirror how examples separate concerns:

| Tier | Typical content | Source |
|------|-----------------|--------|
| `calib` | Economic parameters ($\beta$, taxes, preferences) | `calibration.yaml` + overrides |
| `settings` | Numerics (grid sizes, tolerances, plot flags) | `settings.yaml` + overrides |
| `config` | Anything else; no tier guardrails | Override file + `--config-override` |

**Precedence** for keys that belong to `calib` or `settings`: **CLI > override file > base YAML**. The CLI enforces that `--calib-override` cannot set a key that lives only in `settings`, and vice versa. `--config-override` can shadow YAML keys but emits warnings when it overlaps known calibration or settings keys.

### Modes

| `run.mode` | How it is selected | Typical example behavior |
|------------|--------------------|---------------------------|
| `single` | Default (no `--compare`, no `--sweep`) | One solve; optional simulate/plots |
| `compare` | `--compare A B ...` | Several solves with different methods or per-stage overrides |
| `sweep` | `--sweep` plus grid axes | `param_grid` + `sweep()`; tables to `output_dir` |

`--compare` and `--sweep` are mutually exclusive; `parse_run` errors if both are active.

### Override system

| Mechanism | Effect |
|-----------|--------|
| `--calib-override k=v` | Merged into `run.calib` (tier-checked) |
| `--setting-override k=v` | Merged into `run.settings` (tier-checked) |
| `--config-override k=v` | Merged into `run.config` (warnings if key exists in YAML tiers) |
| `--override-file path.yaml` | Sparse YAML; keys routed to calib/settings/config by name |
| `--method-override stage.scheme=TAG` | Fills `run.method_overrides` (dynx routing) |

Global method: `--method NAME` sets `run.method`; omit it and the YAML default applies (`None` on `RunSpec`).

---

## Quick start

```python
from pathlib import Path
from kikku.run import parse_run

def solve_only(model_dir: Path, calib: dict, settings: dict):
    # Your model's solve entry (illustrative)
    ...

def main():
    run = parse_run(
        name="my_model",
        syntax="path/to/syntax/separable",
        methods=["FUES", "NEGM"],
        modes=["compare", "sweep", "simulate", "plots"],
        output="results/my_model",
    )
    cfg = {**run.settings, **run.config}
    nest, grids = solve_only(run.model_dir, run.calib, cfg)
    # write outputs under run.output_dir
```

`run.output_dir` is created by `make_run_dir` before `RunSpec` is returned (see [cli.md](cli.md)).

---

## How examples use it

FUES examples follow the same skeleton:

1. Call `parse_run(..., modes=[...])` with the example name, model directory, allowed methods, and enabled capabilities (`compare`, `sweep`, `simulate`, `mpi`, `gpu`, `plots` register extra flags).
2. Print or log `run.output_dir`.
3. Branch on `run.mode` and dispatch to `run_single`, `run_comparison`, or `run_sweep` (names vary).

```python
from kikku.run import parse_run

def main():
    run = parse_run(
        name="durables",
        syntax="examples/durables/syntax/separable",
        methods=["FUES", "NEGM"],
        modes=["compare", "sweep", "simulate"],
        output="results/durables",
    )
    print(f"Output directory: {run.output_dir}")

    if run.mode == "compare":
        run_comparison(run)
    elif run.mode == "sweep":
        run_sweep(run)
    else:
        run_single(run)
```

Inside `run_single`, merge tiers the way your solver expects (often `{**run.settings, **run.config}` for the settings overlay). Pass `run.method`, `run.method_overrides`, and `run.calib` into your solve function. For sweeps, build a grid from `run.sweep_params` / `run.sweep_grids` and call `kikku.run.sweep.sweep` as in [sweep.md](sweep.md).

---

## Further reading

- [CLI reference](cli.md) — every `RunSpec` field and flag
- [Persistence](persistence.md) — `save_solution` vs `save_nest`
- [Sweep module](sweep.md) — grids, metrics, MPI
