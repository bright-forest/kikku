# Running SMM Estimation on NCI Gadi

## Overview

This guide explains how to set up and run a Simulated Method of Moments (SMM) estimation job on NCI's Gadi supercomputer using the kikku estimation infrastructure. The infrastructure works with **any model** that can produce simulated lifecycle panels — it does not require dolo+ or the DDSL pipeline.

## What you need

### 1. A trial function

A Python function that takes a parameter dict and returns simulated data:

```python
def trial(theta):
    """
    theta: dict[str, float]  e.g. {'beta': 0.95, 'sigma': 2.0}
    returns: dict[str, np.ndarray]  e.g. {'c': (T, N), 'a': (T, N), ...}
    """
    # Your model here — solve, simulate, return panels
    model = solve_my_model(theta)
    panels = simulate(model, N=10000, seed=sim_seed)
    return panels
```

**Requirements on the return value:**
- Dict of `{variable_name: (T, N) numpy array}`
- T = time periods (rows), N = agents (columns)
- Row index typically = age or period
- Missing data as `np.nan`
- Discrete variables as numeric (0, 1, ...), not strings

### 2. An estimation YAML

Lives inside your model directory. Specifies free parameters, moment targets, CE settings, and output paths:

```yaml
estimation:
  free:
    beta:
      bounds: [0.88, 0.99]
    sigma:
      bounds: [1.0, 5.0]

  method: cross-entropy
  method_options:
    n_samples: 520        # candidates per CE iteration
    n_elite: 20           # elite set size
    max_iter: 50          # max CE iterations
    tol: 1e-3             # convergence tolerance
    sampling_seed: 42     # RNG for parameter draws
    simulation_seed: 99   # RNG for model simulation (CRN)

  scratch_dir: /scratch/tp66/{user}/my_estimation
  results_dir: results/my_model/estimation

moments:
  data_source: precomputed          # or selfgen
  data_file: moments_data.csv

  setup:
    t0: 0                           # 0 if panel row index = age
    T: 70
    age_groups:
      1: [20, 24]
      2: [25, 29]
      3: [30, 34]
      # ... etc

  identification:
    mean: [c, a]
    sd: [c, a]
    corrs: [[c, a]]
    autocorrs: [c, a]
```

### 3. An estimate.py entry point

A script that wires your trial function to the kikku estimation module:

```python
"""estimate.py — SMM entry point for my_model."""

import argparse
import os
import yaml
from pathlib import Path

from kikku.run.estimate import load_estimation_spec, make_criterion, estimate, diagnostics
from kikku.run.moments import make_moment_fn, moment_names as get_moment_names
from kikku.run.mpi import get_comm, is_root

# Your model's solve and simulate
from my_model import solve_model, simulate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, help='Path to estimation YAML')
    parser.add_argument('--scratch', default=None)
    parser.add_argument('--results', default=None)
    parser.add_argument('--N-sim', type=int, default=10000)
    args = parser.parse_args()

    comm = get_comm()
    spec = load_estimation_spec(args.spec)

    moment_fn = make_moment_fn(spec['moment_spec'])
    simulation_seed = int(spec['method_options'].get('simulation_seed', 99))

    # --- Data moments ---
    if spec['moment_spec'].get('data_source') == 'selfgen':
        # Generate data at default params
        data_panels = simulate_model(default_params, N=args.N_sim, seed=simulation_seed)
        data_moments = moment_fn(data_panels)
    else:
        data_moments = spec['data_moments']

    # --- Trial function ---
    def trial(theta):
        model = solve_model(theta)
        return simulate_model(model, N=args.N_sim, seed=simulation_seed)

    # --- Estimate ---
    criterion = make_criterion(trial, moment_fn, data_moments)
    result = estimate(
        criterion, spec['free'],
        method=spec['method'],
        method_options=spec['method_options'],
        comm=comm,
        verbose=is_root(comm),
    )

    # --- Report ---
    if is_root(comm):
        print(f"theta*: {result.theta}")
        print(f"Loss: {result.objective:.6f}")
        diag = diagnostics(result, data_moments,
                           moment_names=get_moment_names(spec['moment_spec']))
        for row in diag['worst_moments']:
            print(f"  {row['moment']:40s} contrib={row['contribution']:.4f}")


if __name__ == '__main__':
    main()
```

## PBS script template

### Queue selection

| n_samples | Queue | Nodes | ncpus |
|-----------|-------|-------|-------|
| ≤ 104 | normalsr | 1 | 104 |
| 208 | normalsr | 2 | 208 |
| 520 | normalsr | 5 | 520 |
| 1040 | normalsr | 10 | 1040 |

Rule: `ncpus = n_samples`, rounded up to the nearest multiple of 104 (normalsr cores/node). Each core evaluates one candidate per CE iteration.

### Template

```bash
#!/bin/bash
#PBS -P tp66
#PBS -q normalsr
#PBS -N my_est
#PBS -l ncpus=520
#PBS -l mem=2500GB
#PBS -l walltime=5:00:00
#PBS -l jobfs=200GB
#PBS -l storage=scratch/tp66+gdata/tp66
#PBS -l wd
#PBS -o /g/data/tp66/logs/my_model/
#PBS -e /g/data/tp66/logs/my_model/

module purge
module load python3/3.11.0
module load openmpi/4.1.5

# --- Venv: create if missing, update packages ---
VENV=/scratch/tp66/$USER/venvs/my_model

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV..."
    python3 -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"

pip install -e "$PBS_O_WORKDIR" --quiet 2>&1 | tail -1
pip install -e "git+https://github.com/bright-forest/kikku.git#egg=kikku[estimation]" --quiet 2>&1 | tail -1
pip install --no-binary :all: mpi4py --quiet 2>&1 | tail -1

# --- Thread pinning (critical for MPI Python) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Numba cache on local SSD
export NUMBA_CACHE_DIR=$PBS_JOBFS/numba_cache
mkdir -p $NUMBA_CACHE_DIR

# Suppress OpenMPI warnings
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll=^hcoll
export PMIX_MCA_gds=hash

# --- Run ---
echo "Starting at $(date), $PBS_NCPUS cores"

mpiexec -n $PBS_NCPUS \
    --map-by ppr:13:numa \
    python3 -u -m mpi4py estimate.py \
        --spec path/to/estimation/baseline.yaml \
        --scratch /scratch/tp66/$USER/my_estimation \
        --results results/my_model \
        --N-sim 10000

echo "Finished at $(date)"

# --- Move logs to dated subfolder ---
LOG_DIR=/g/data/tp66/logs/my_model/$(date +%Y-%m-%d)
mkdir -p "$LOG_DIR"
mv /g/data/tp66/logs/my_model/${PBS_JOBNAME}.o${PBS_JOBID%%.*} "$LOG_DIR/" 2>/dev/null
mv /g/data/tp66/logs/my_model/${PBS_JOBNAME}.e${PBS_JOBID%%.*} "$LOG_DIR/" 2>/dev/null
```

## How the cross-entropy loop works

```
Iteration 0:  draw n_samples uniformly within bounds
              ↓
              evaluate trial(theta) for each candidate (MPI parallel)
              ↓
              sort by loss, select top n_elite
              ↓
              compute weighted mean + covariance of elite set

Iteration 1+: draw n_samples from truncated MVN(mean, cov)
              ↓
              evaluate, select, update (same as above)
              ↓
              repeat until |change in elite mean loss| < tol
```

Each MPI rank evaluates one candidate independently. Rank 0 coordinates the draws, gathers losses, and updates the distribution. All ranks use the same `simulation_seed` for CRN (common random numbers).

## Two seeds, two purposes

```yaml
sampling_seed: 42     # controls which thetas are explored
simulation_seed: 99   # controls shock draws inside simulate (same for all thetas)
```

- **sampling_seed**: change this to run independent estimation attempts (different exploration of parameter space)
- **simulation_seed**: change this to test sensitivity to simulation noise

Same seeds + same n_samples = identical estimation path, regardless of number of MPI ranks.

## Moment matching: how it works

The moment function and data loader both produce `dict[str, float]`. Matching is by key:

```python
moment_fn = make_moment_fn(spec)     # panels → {'mean_c__age5': 0.34, ...}
data_moments = load_data_moments(csv) # CSV → {'mean_c__age5': 0.31, ...}
# Loss = sum of squared differences over common keys
```

### Defining moments (YAML)

**Bulk generation** — declare types × variables × age groups:
```yaml
identification:
  mean: [c, a, h]
  sd: [c, a]
  corrs: [[c, a], [a, h]]
  autocorrs: [c, a]
```
Produces: `mean_c__age1`, `sd_a__age3`, `corr_c_a__age5`, `autocorr_c__age2`, etc.

**Explicit targets** — match a specific CSV column name:
```yaml
targets:
  - key: av_consumption_14_0    # CSV column name
    stat: mean
    var: c                      # sim variable to compute mean of
```

**Conditional moments** — by discrete branch:
```yaml
conditional:
  discrete:
    0: {mean: [c, a]}          # mean c and a for branch 0
    1: {mean: [c, h]}          # mean c and h for branch 1
```

### Three data sources

| Source | YAML setting | How it works |
|--------|-------------|-------------|
| Pre-computed CSV | `data_source: precomputed` | Load scalars from CSV, key by `{column}__age{row}` |
| Self-generated | `data_source: selfgen` | Run model at default params, apply same `moment_fn` |
| Raw microdata | `data_source: microdata` | Load panel DataFrame, compute moments from it |

## Saving and loading nests

For models using the DDSL pipeline, save/load the full solved nest:

```python
from kikku.run.nest_io import save_nest, load_nest, nest_info

# Save (with solutions — ~25MB at 300-grid)
save_nest(nest, 'results/best_model.nst',
          metadata={'theta': result.theta, 'loss': result.objective})

# Save (without solutions — ~80KB, just the configured period objects)
save_nest(nest, 'results/configured.nst', solutions=False)

# Load
nest = load_nest('results/best_model.nst')
# nest['periods'][0]['stages']['keeper_cons'] → dolo+ SymbolicModel
# nest['solutions'] → solution arrays (if saved)
# nest['metadata'] → {'theta': ..., 'loss': ...}

# Quick info
nest_info('results/best_model.nst')
# → {n_periods: 31, has_solutions: True, file_size_mb: 25.4, ...}
```

For non-DDSL models, use `save_solution` / `load_solution` for numpy arrays:

```python
from kikku.run.io import save_solution, load_solution

save_solution('results/my_solution/', solution_dict, metadata={'theta': ...})
sol, meta = load_solution('results/my_solution/')
```

## Cost estimation

```
SU = rate × ncpus × walltime_hours
```

| Queue | Rate | 520 cores × 5h | 1040 cores × 5h |
|-------|------|-----------------|------------------|
| normalsr | 2.0 | 5,200 SU | 10,400 SU |
| express | 6.0 | 15,600 SU | 31,200 SU |

Most estimation runs converge well within the walltime budget. Early convergence saves SU.

## Checklist before submitting

1. `n_samples` in the YAML matches `ncpus` in the PBS script
2. `ncpus` is a multiple of 104 (normalsr) or 48 (normal)
3. `#PBS -l storage=` lists every `/scratch/` and `/g/data/` project you access
4. `OMP_NUM_THREADS=1` is set (prevents numpy thread oversubscription)
5. `mpi4py` is built from source (`pip install --no-binary :all: mpi4py`)
6. The venv is on `/scratch/`, not `$HOME` (10 GiB quota)
7. `scratch_dir` and `results_dir` in the YAML point to different filesystems
8. Log directories exist: `mkdir -p /g/data/tp66/logs/my_model/`
