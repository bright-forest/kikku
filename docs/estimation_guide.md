# Kikku Estimation Module: User Guide

## Overview

The estimation module lets you estimate structural parameters by matching simulated moments to data moments (SMM/MSM). It lives in `kikku.run` and consists of two files:

- `moments.py` — computes named moments from panel data
- `estimate.py` — composes the SMM criterion and runs optimisation

The pipeline:

```
theta  -->  trial(theta)  -->  moment_fn(panels)  -->  loss(sim, data)  -->  scalar
  ^                                                                            |
  |_________________ optimiser (CE or scipy-DE) ______________________________|
```

You provide a **trial function** that takes parameters and returns simulated panels. The module generates the moment function from a YAML spec and handles everything else.

## What you need to provide

### 1. A trial function: `theta_dict -> sim_panels`

This is the only model-specific code. It takes a dict of parameter values and returns simulated lifecycle panels.

```python
def trial(theta):
    """theta is {'beta': 0.95, 'alpha': 0.7, ...} -> sim_panels dict."""
    nest, grids = solve(
        syntax_dir,
        calib_overrides=theta,
        setting_overrides={'n_a': 200, 'n_h': 200},
        verbose=False,
    )
    sim_data = simulate_lifecycle(nest, grids, N=10000, seed=42)
    return sim_data
```

**Requirements on sim_panels:**
- A dict of `{variable_name: (T, N) numpy array}`
- `T` = number of time periods (rows), `N` = number of agents (columns)
- Row index = age (or period). The moment spec's `t0` controls how age groups map to row indices.
- Missing data as `np.nan`
- Discrete choices (e.g. `discrete`, `z_idx`) should be numeric (0, 1, ...), not strings

For durables2_0, `simulate_lifecycle` returns exactly this format:
```python
sim_panels = {
    'a': (70, 10000),       # financial assets
    'h': (70, 10000),       # housing stock
    'c': (70, 10000),       # consumption
    'y': (70, 10000),       # income
    'z_idx': (70, 10000),   # wage shock index
    'discrete': (70, 10000),# 0=keeper, 1=adjuster
    'a_nxt': (70, 10000),   # end-of-period assets
    'h_nxt': (70, 10000),   # end-of-period housing
}
```

### 2. A moment specification YAML

Defines which statistics to compute and how to match them to data. Example:

```yaml
estimation:
  free:
    beta:
      bounds: [0.88, 0.99]
      init: 0.945
    alpha:
      bounds: [0.3, 0.95]
      init: 0.7

  method: cross-entropy
  method_options:
    n_samples: 48
    n_elite: 10
    max_iter: 50
    tol: 1e-3
    seed: 42

moments:
  data_source: selfgen          # selfgen | precomputed
  data_file: moments_data.csv   # only used when precomputed

  setup:
    t0: 0
    T: 70
    age_groups:
      5: [40, 44]
      6: [45, 49]
      7: [50, 54]

  identification:
    mean: [c, a, h]
    sd: [c, a]
    corrs: [[c, h], [a, h]]
    autocorrs: [c, a, h]
    branch_fractions: [discrete]
    conditional:
      discrete:
        0: {mean: [c, a]}
        1: {mean: [c, h]}
```

### 3. Data moments (one of three sources)

**Scenario 1 — Self-generated (recovery test):**
Set `data_source: selfgen`. Simulate the model at known true parameters, apply the same `moment_fn` to get data moments. Keys match by construction.

**Scenario 2 — Pre-computed CSV:**
Set `data_source: precomputed`. The CSV has age groups as rows, named moments as columns. Keys are `{column_name}__age{row}`.

**Scenario 3 — Raw microdata:**
Set `data_source: microdata`. Module loads the file, converts to panels, computes moments using the same spec.

## How the pieces compose

```python
import yaml
from kikku.run.moments import make_moment_fn, load_data_moments
from kikku.run.estimate import make_criterion, estimate, diagnostics

# 1. Load spec
with open('estimation/baseline.yaml') as f:
    spec = yaml.safe_load(f)

# 2. Build moment function (YAML -> callable)
moment_fn = make_moment_fn(spec['moments'])
#   moment_fn: sim_panels dict -> dict[str, float]
#   e.g. {'mean_c__age5': 0.342, 'sd_a__age6': 1.205, ...}

# 3. Get data moments
# Option A: self-generated
data_moments = moment_fn(trial(theta_true))
# Option B: from CSV
data_moments = load_data_moments('moments_data.csv', spec['moments'])

# 4. Compose the criterion
criterion = make_criterion(trial, moment_fn, data_moments)
#   criterion: theta_dict -> scalar loss

# 5. Estimate
result = estimate(criterion, spec['estimation']['free'],
                  method='cross-entropy',
                  method_options=spec['estimation']['method_options'])

# 6. Diagnostics
diag = diagnostics(result, data_moments)
```

## What `make_criterion` does internally

1. Sorts `data_moments` keys alphabetically → deterministic ordering
2. Builds `data_vec` from sorted keys
3. Returns closure `criterion(theta)` that:
   - Calls `trial_fn(theta)` → sim_panels
   - Calls `moment_fn(sim_panels)` → sim_dict
   - Builds `sim_vec` aligned to same sorted keys (NaN for missing)
   - Returns `(sim - data)' W (sim - data) + 1e6 * n_NaN`
   - On exception: returns `1e10`

## How moment matching works

### Dict keys are the contract

Both `moment_fn(panels)` and `load_data_moments(csv)` return `dict[str, float]`. Keys must match.

**Bulk generation** (from `identification:`) produces:
```
mean_c__age5, sd_a__age6, corr_c_h__age7, autocorr_a__age5,
branch_frac_discrete_1__age5, cond_discrete_0_mean_c__age6
```

**Targets** (explicit mapping to CSV columns):
```yaml
targets:
  - key: av_consumption2_14_0
    stat: mean
    var: c
```
Produces: `av_consumption2_14_0__age5`, etc.

| Data source | How to match |
|---|---|
| Self-generated | Same `moment_fn` on both → identical keys |
| CSV + `identification:` | Keys differ unless column names = canonical names. Use `targets:` instead. |
| CSV + `targets:` | Target `key` matches CSV column. Both get `__age{group}` suffix. |

### Age groups

```yaml
age_groups:
  5: [40, 44]     # group 5 = array rows 40, 41, 42, 43
```

Set `t0: 0` if panel row index = age. Set `t0: 20` if row 0 = age 20. Formula: `array_row = age - t0`.

## The cross-entropy optimiser

1. Draw `n_samples` from truncated MVN
2. Evaluate `criterion(theta)` for each (via `mpi_map` if MPI)
3. Select top `n_elite`, compute weighted mean+cov
4. Broadcast, repeat until convergence

```bash
# Serial
python3 estimate.py

# Parallel (one candidate per core)
mpirun -np 48 python3 -m mpi4py estimate.py
```

Pass `comm=get_comm()` to `estimate()`.

## Complete example: durables2_0

```python
from examples.durables2_0.solve import solve
from examples.durables2_0.simulate import simulate_lifecycle
from kikku.run.moments import make_moment_fn
from kikku.run.estimate import make_criterion, estimate, diagnostics
from kikku.run.mpi import get_comm
import yaml

with open('examples/durables2_0/estimation/baseline.yaml') as f:
    spec = yaml.safe_load(f)

moment_fn = make_moment_fn(spec['moments'])

def trial(theta):
    nest, grids = solve('examples/durables2_0/syntax',
                        calib_overrides=theta,
                        setting_overrides={'n_a': 200, 'n_h': 200, 'n_w': 200},
                        verbose=False)
    return simulate_lifecycle(nest, grids, N=10000, seed=42)

# Self-generated data
theta_true = {'beta': 0.945, 'alpha': 0.7, 'gamma_c': 3.5, 'tau': 0.12}
data_moments = moment_fn(trial(theta_true))

# Estimate
criterion = make_criterion(trial, moment_fn, data_moments)
result = estimate(criterion, spec['estimation']['free'],
                  method='cross-entropy',
                  method_options=spec['estimation']['method_options'],
                  comm=get_comm())

diag = diagnostics(result, data_moments)
print(f"theta*: {result.theta}")
print(f"loss:   {result.objective:.6f}")
```

## Derived variables

```yaml
setup:
  derived_variables:
    total_wealth: a + h
    log_c: np.log(np.maximum(c, 1e-10))
```

Computed before moment extraction. Safe evaluator (no `eval()`).

## Conditional moments

```yaml
identification:
  conditional:
    discrete:
      0: {mean: [c, a]}
      1: {mean: [c, h], corrs: [[c, h]]}
  branch_fractions: [discrete]
```

## Utility switching

```python
solve('examples/durables2_0/syntax', ...)       # separable CRRA
solve('examples/durables2_0/syntax_cd', ...)     # Cobb-Douglas CRRA
```

Same sim_panels format. Same moment function. Different callables dispatched automatically.
