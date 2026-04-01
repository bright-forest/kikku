# Parameter Sweeps (`kikku.run.sweep`)

We often want to solve a model across a grid of configurations — varying grid resolution, calibration parameters, or solution methods — and collect timing and accuracy metrics from each run. The sweep module provides three tools for this: `param_grid` constructs the design matrix, `sweep` executes it (with optional best-of-$n$ and MPI distribution), and `format_table` / `write_table` render the results.

The sweep module is model-agnostic. It does not know about calibration, settings, or stages — the user's `solve_fn` closure handles all tier routing internally.

---

## Building the parameter grid

`param_grid` takes named keyword arguments, each a list of values, and returns their Cartesian product as a list of override dicts:

```python
from kikku.run.sweep import param_grid

# One axis: grid convergence
grid = param_grid(n_a=[100, 200, 300, 500])
# [{'n_a': 100}, {'n_a': 200}, {'n_a': 300}, {'n_a': 500}]

# Two axes: grid size × transaction cost
grid = param_grid(n_a=[100, 300], tau=[0.06, 0.12])
# [{'n_a': 100, 'tau': 0.06},
#  {'n_a': 100, 'tau': 0.12},
#  {'n_a': 300, 'tau': 0.06},
#  {'n_a': 300, 'tau': 0.12}]
```

Each dict is handed to `solve_fn` as-is. The sweep does not know which keys are calibration parameters and which are numerical settings — the `solve_fn` closure routes them to the correct tier.

---

## Running a sweep

```python
results = sweep(solve_fn, param_grid, metric_fns,
                n_reps=3, warmup=True, best='min')
```

### Arguments

| Argument | Type | Default | Role |
|----------|------|---------|------|
| `solve_fn` | `dict -> Any` | — | Takes one override dict, returns an opaque result |
| `param_grid` | `list[dict]` | — | From `param_grid()` |
| `metric_fns` | `dict[str, result -> float]` | — | Named extractors applied to each result. **First key is the primary metric** for best-of-$n$ selection |
| `n_reps` | `int` | 3 | Repetitions per point (best rep kept) |
| `warmup` | `bool` | True | Run `solve_fn` once before timing (result discarded) |
| `best` | `'min'` or `'max'` | `'min'` | Whether lower or higher primary metric is better |
| `on_error` | `'raise'` or `'skip'` | `'raise'` | `'skip'` fills NaN and continues |
| `comm` | MPI communicator | None | Distributes points across ranks |

### What happens at each point

For each override dict `ov` in the grid:

1. **Warmup.** Call `solve_fn(ov)` once, discard the result. This absorbs JIT compilation or file-loading overhead so it does not contaminate the timing runs.
2. **Repetitions.** Call `solve_fn(ov)` $n$ times. After each call, evaluate every function in `metric_fns` on the result.
3. **Selection.** Keep the rep whose primary metric (the first key in `metric_fns`) is best — lowest if `best='min'`, highest if `best='max'`. All metrics from that rep are retained.
4. **Emit.** Merge the override dict with the best-rep metrics: `{**ov, **best_metrics}`.

### Return value

A flat `list[dict]`, one dict per grid point. Each dict merges the override keys from `param_grid` with the scalar values produced by `metric_fns`:

```
{**override_dict, **best_rep_metrics}
```

Concretely, if the grid has axis `n_a` and the metric functions are `solve_ms`, `keeper_ms`, `adj_ms`, the output is:

```python
[
    {'n_a': 100, 'solve_ms':  120.3, 'keeper_ms':  15.2, 'adj_ms':  5.1},
    {'n_a': 200, 'solve_ms':  340.6, 'keeper_ms':  42.3, 'adj_ms': 12.7},
    {'n_a': 300, 'solve_ms':  780.1, 'keeper_ms':  95.4, 'adj_ms': 28.3},
    {'n_a': 500, 'solve_ms': 2100.5, 'keeper_ms': 260.1, 'adj_ms': 72.0},
]
```

The override keys (`n_a`) identify the configuration. The metric keys (`solve_ms`, `keeper_ms`, `adj_ms`) are the best-rep scalar values. When `n_reps > 1`, only the metrics from the single best rep (selected by the primary metric) appear — not averages, not all reps. When a point fails with `on_error='skip'`, all metric values are `NaN`.

This flat structure feeds directly into `format_table` and `write_table`.

---

## Example 1: Timing sweep

The standard grid-convergence experiment. We solve at several grid sizes, take the fastest of three runs, and report per-stage timing.

```python
from kikku.run.sweep import param_grid, sweep
from kikku.run.metrics import format_table, write_table
from my_model.solve import solve
from my_model.outputs import get_timing

grid = param_grid(n_a=[100, 200, 300, 500])

def solve_fn(ov):
    nest, grids = solve('syntax/', setting_overrides={'n_a': ov['n_a']})
    return nest

metric_fns = {
    'solve_ms':  lambda r: get_timing(r)['solve_time'] * 1000,
    'keeper_ms': lambda r: get_timing(r)['keeper_ms'],
    'adj_ms':    lambda r: get_timing(r)['adj_ms'],
}

results = sweep(solve_fn, grid, metric_fns, n_reps=3, best='min')

cols = ['n_a', 'solve_ms', 'keeper_ms', 'adj_ms']
print(format_table(results, cols))
write_table('results/tables/timing_sweep.md', results, cols)
```

Notice that `solve_fn` routes `ov['n_a']` to `setting_overrides` — the sweep itself has no opinion about what `n_a` is.

---

## Example 2: Sweep with simulation and Euler errors

When forward simulation dominates the cost, best-of-$n$ timing is meaningless. We set `n_reps=1` and compose solve + simulate + Euler evaluation into a single `trial_fn`:

```python
grid = param_grid(n_a=[100, 200, 300])

def trial_fn(ov):
    nest, grids = solve('syntax/', setting_overrides={'n_a': ov['n_a']})
    sim = simulate_lifecycle(nest, grids, N=10_000, seed=42)
    euler_c = evaluate_euler_c(sim, nest, grids)
    euler_h = evaluate_euler_h(sim, nest, grids)
    timing = get_timing(nest)
    d = sim['discrete']
    return {
        'solve_ms':  timing['solve_time'] * 1000,
        'euler_c':   compute_euler_stats(euler_c, d)['combined']['mean'],
        'euler_h':   compute_euler_stats(euler_h, d)['combined']['mean'],
        'adj_rate':  np.mean(d[d >= 0]) * 100,
    }

# metric_fns are trivial key lookups — all computation is inside trial_fn
metric_fns = {k: (lambda k=k: lambda r: r[k])() for k in
              ['solve_ms', 'euler_c', 'euler_h', 'adj_rate']}

results = sweep(trial_fn, grid, metric_fns, n_reps=1, warmup=False)

cols = ['n_a', 'solve_ms', 'euler_c', 'euler_h', 'adj_rate']
print(format_table(results, cols))
```

The `trial_fn` pattern keeps all model logic in one closure. The sweep sees only `dict -> dict`.

---

## Example 3: Multi-dimensional sweep

Sweep over grid size and a calibration parameter simultaneously:

```python
grid = param_grid(
    n_a=[100, 200, 300],
    tau=[0.06, 0.09, 0.12],
)
# 9 points: 3 × 3

def solve_fn(ov):
    nest, grids = solve('syntax/',
                        setting_overrides={'n_a': ov['n_a']},
                        calib_overrides={'tau': ov['tau']})
    return nest

results = sweep(solve_fn, grid, metric_fns, n_reps=3)
# Each row: {'n_a': 100, 'tau': 0.06, 'solve_ms': ..., ...}
```

The `solve_fn` decides that `n_a` is a setting and `tau` is a calibration parameter. The sweep is indifferent.

---

## MPI distribution

Pass an MPI communicator to distribute grid points across ranks. Each rank solves its slice; results are gathered to rank 0.

```python
from kikku.run.mpi import get_comm

comm = get_comm()  # None if not in mpirun
results = sweep(solve_fn, grid, metric_fns, comm=comm)
# rank 0: full results list
# other ranks: []
```

From the command line:

```bash
mpirun -np 8 python -m mpi4py -m examples.durables.run \
    --sweep --sweep-grids 100,200,300,500,1000
```

When `mpi4py` is not installed or the script is not launched via `mpirun`, `get_comm()` returns `None` and the sweep runs serially. No code changes needed.

**Output directories under MPI.** `make_run_dir` is MPI-safe: rank 0 creates the `YYYY-MM-DD/NNN/` directory and broadcasts the path to all ranks. Every rank uses the same folder — no racing or duplicate directories.

---

## Formatting and saving results

```python
from kikku.run.metrics import format_table, write_table

# Markdown table (default)
print(format_table(results, cols))

# Fewer decimal places
print(format_table(results, cols, float_fmt='.2f'))

# LaTeX with caption
tex = format_table(results, cols, fmt='latex', caption='Grid convergence')

# Write to disk (parent directories created automatically)
write_table('results/tables/sweep.md', results, cols)
write_table('results/tables/sweep.tex', results, cols,
            fmt='latex', caption='Grid convergence')
```

---

## Error handling

```python
results = sweep(solve_fn, grid, metric_fns, on_error='skip')
```

When `on_error='skip'`, a failed `solve_fn` call fills all metrics with NaN and prints a warning. The sweep continues to the next point. This is useful for large sweeps on HPC where a few parameter combinations may hit numerical issues.

---

## API reference

| Function | Signature | Returns |
|----------|-----------|---------|
| `param_grid` | `(**axes: list) -> list[dict]` | Cartesian product of named axes |
| `sweep` | `(solve_fn, grid, metric_fns, ...) -> list[dict]` | `{**overrides, **best_metrics}` per point |
| `format_table` | `(rows, cols, fmt='markdown', float_fmt='.4f') -> str` | Formatted table string |
| `write_table` | `(path, rows, cols, fmt='markdown') -> None` | Writes formatted table to file |
