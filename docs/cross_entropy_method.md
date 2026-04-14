# Cross-Entropy Method: Implementation Details

## The algorithm

The cross-entropy (CE) method is a derivative-free global optimiser that
maintains a sampling distribution over the parameter space and iteratively
narrows it toward the optimum. It is the sole optimisation method in
`kikku.run.estimate`.

Reference: Kroese, Porotsky & Rubinstein (2006), "The cross-entropy method
for continuous multi-extremal optimization."

## Mathematical description

Given a criterion function `Q(θ)` to minimise over a bounded parameter
space `Θ = [lo₁, hi₁] × ... × [lo_P, hi_P]`:

```
Initialise: t = 0

Iteration t = 0:
    Draw θ₁, ..., θ_S  uniformly from Θ
    Evaluate Lᵢ = Q(θᵢ) for each candidate
    Sort by loss: L_(1) ≤ L_(2) ≤ ... ≤ L_(S)
    Select elite set: E = {θ_(1), ..., θ_(K)}    (K = n_elite)
    Compute weighted mean and covariance from E
    → μ₁, Σ₁

Iteration t ≥ 1:
    Draw θ₁, ..., θ_S  from  N(μ_t, Σ_t)  truncated to Θ
    Evaluate, sort, select elite, update μ_{t+1}, Σ_{t+1}

Converge when |mean(L_elite(t)) - mean(L_elite(t-1))| < tol
```

### Why uniform first?

Iteration 0 uses uniform draws (not MVN) because there is no prior
information about where the optimum lies. Starting from a MVN centred
at an arbitrary `init` point biases the search. The uniform-first pattern
is used in Eggsandbaskets and fempres.

After iteration 0, the elite set provides an empirical estimate of the
promising region, and the MVN distribution concentrates draws there.

## Code walkthrough

### Entry point: `estimate()`

```python
from kikku.run.estimate import estimate

result = estimate(criterion, param_spec,
                  method='cross-entropy',
                  method_options={...},
                  comm=comm)
```

This dispatches to `_cross_entropy_minimize()`. The `criterion` is a
callable `theta_dict → float` (composed by `make_criterion`).

### Configuration (from YAML `method_options`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_samples` | int | 48 | Candidates drawn per iteration (= MPI ranks) |
| `n_elite` | int | 10 | Size of the elite set (top-K by loss) |
| `max_iter` | int | 50 | Maximum CE iterations |
| `tol` | float | 1e-2 | Convergence tolerance on elite mean loss change |
| `sampling_seed` | int | 0 | RNG seed for parameter draws |
| `simulation_seed` | int | 99 | RNG seed for model simulation (CRN) |
| `checkpoint_dir` | str | None | Directory for per-iteration state checkpoints |

### Step-by-step code flow

#### 1. Initialisation

```python
names = sorted(param_spec.keys())      # deterministic parameter ordering
rng = np.random.default_rng(sampling_seed)  # single RNG, advanced each iteration
best_theta = midpoint of bounds        # initial best (never evaluated, just a placeholder)
best_loss = 1e10                       # BIG_LOSS
means = None                           # no distribution yet (triggers uniform on iter 0)
cov = None
```

The RNG is created once from `sampling_seed` and carries state across all
iterations. This means the same seed always produces the same sequence of
candidates regardless of how many iterations run.

#### 2. Sampling

```python
# Rank 0 only:
if it == 0 or means is None:
    candidates = _sample_uniform(param_spec, n_samples, rng)
else:
    candidates = _sample_bounded(means, cov, param_spec, n_samples, rng)

# Broadcast to all ranks:
candidates = bcast_item(candidates, comm, root=0)
```

**`_sample_uniform`**: draws each parameter independently from
`Uniform(lo, hi)`. No rejection needed — all draws are in-bounds by
construction.

**`_sample_bounded`**: draws from `MVN(mean, cov)` with rejection
sampling. If a draw falls outside bounds, it is re-drawn (up to 1000 retries).
If all retries fail, the draw is clipped to bounds. This handles the case
where the MVN concentrates near a boundary.

Both functions advance the same `rng` object, maintaining deterministic
sequencing across iterations.

#### 3. Parallel evaluation

```python
losses = mpi_map(lambda th: _safe_criterion(criterion, th), candidates, comm=comm)
```

`mpi_map` distributes candidates across MPI ranks via `scatter_items`,
evaluates the criterion on each rank's slice, and gathers results to rank 0.
In serial mode (`comm=None`), this is a list comprehension.

`_safe_criterion` wraps the criterion call in a try/except — if the model
solver crashes for a particular theta, it returns `BIG_LOSS = 1e10` instead
of killing the MPI job. This is critical for robustness on HPC.

#### 4. Elite selection

```python
paired = sorted(zip(candidates, losses), key=lambda p: p[1])
elite_pairs = paired[:n_elite]
elite_thetas = [p[0] for p in elite_pairs]
elite_losses = [float(p[1]) for p in elite_pairs]
```

Sort all candidates by loss (ascending — lower is better). Take the top
`n_elite`. This is the standard CE elite selection.

#### 5. Best tracking

```python
for th, ell in zip(elite_thetas, elite_losses):
    if ell < best_loss:
        best_loss = ell
        best_theta = dict(th)
```

Track the global best theta across all iterations (not just the current
iteration's elite). This is the value returned as `result.theta`.

#### 6. Distribution update

```python
mean_vec, cov = _elite_weighted_mean_cov(elite_thetas, elite_losses, names)
means = _dict_from_vector(mean_vec, names)
```

**`_elite_weighted_mean_cov`** computes exponential weights:

```python
w_i ∝ exp(-(L_i - min(L)))
```

This gives the best candidate weight 1.0 and down-weights worse candidates
exponentially. The weighted mean and covariance are:

```
μ = Σ wᵢ θᵢ
Σ = Σ wᵢ (θᵢ - μ)(θᵢ - μ)ᵀ + εI
```

where `ε = 1e-12` is a regularisation term preventing singular covariance.

This weighting scheme follows fempres. The standard CE uses uniform weights
over the elite set; exponential weights give more influence to better
candidates, which can improve convergence.

#### 7. Convergence check

```python
if elite_mean_loss_prev is not None:
    if abs(elite_mean_loss_prev - elite_mean_loss) < tol:
        converged = True
```

Convergence is declared when the mean loss of the elite set changes by less
than `tol` between consecutive iterations. This means the distribution has
stabilised — the elite are no longer improving significantly.

Note: this checks the *elite mean*, not the *best* loss. The best loss may
still improve slightly, but if the elite mean is stable, the distribution
has converged.

#### 8. Broadcast updated state

```python
converged = bcast_item(converged, comm, root=0)
means = bcast_item(means, comm, root=0)
cov = bcast_item(cov, comm, root=0)
best_theta = bcast_item(best_theta, comm, root=0)
best_loss = bcast_item(best_loss, comm, root=0)
history = bcast_item(history, comm, root=0)
```

After the elite update on rank 0, the new distribution and convergence
flag are broadcast to all ranks. `bcast_item` degrades to identity when
`comm=None` (serial mode).

This synchronisation point ensures all ranks have the same state before
the next iteration's sampling step.

#### 9. Final evaluation

```python
_safe_criterion(criterion, best_theta)
sim_moments = dict(getattr(criterion, "last_sim_moments", None) or {})
```

After convergence, the criterion is evaluated one more time at `best_theta`
to populate `criterion.last_sim_moments` — the simulated moments at the
optimum. These are stored in the `EstimationResult` for diagnostics.

### Return value: `EstimationResult`

```python
@dataclass
class EstimationResult:
    theta: dict[str, float]              # best parameters found
    objective: float                      # loss at theta
    converged: bool                       # did elite mean loss stabilise?
    n_iter: int                           # number of CE iterations run
    history: list[dict[str, Any]]         # per-iteration: means, cov, best_loss, elite_mean_loss
    sim_moments: dict[str, float]         # simulated moments at theta*
```

## MPI topology

```
Rank 0 (coordinator):
    ├── draws candidates (uniform or MVN)
    ├── broadcasts candidate list to all ranks
    ├── receives losses from all ranks (via mpi_map gather)
    ├── sorts, selects elite, updates distribution
    ├── broadcasts updated means/cov/convergence
    └── writes checkpoints (if checkpoint_dir set)

Ranks 1..S-1 (workers):
    ├── receive candidate list (via bcast)
    ├── evaluate their slice of candidates (via mpi_map scatter)
    ├── return losses (via mpi_map gather)
    └── receive updated state (via bcast)
```

All ranks participate in `mpi_map` which internally uses `scatter_items`
and `gather_results`. Rank 0 does extra work (sorting, updating) but this
is O(n_samples × n_params) — negligible compared to the model evaluations.

## Failure handling

| Failure | What happens | Where |
|---------|-------------|-------|
| Model solver crashes for one theta | `_safe_criterion` returns `BIG_LOSS = 1e10` | `_safe_criterion` |
| All retries in rejection sampling fail | Draw is clipped to bounds | `_sample_bounded` |
| Covariance becomes singular | `ε = 1e-12` regularisation added to diagonal | `_elite_weighted_mean_cov` |
| Elite weights sum to zero | Fall back to uniform weights | `_elite_weighted_mean_cov` |
| MPI rank dies | Job crashes (no rank-level fault tolerance) | — |

## Checkpointing

When `checkpoint_dir` is set, rank 0 saves a `state.pkl` after each
iteration containing:

```python
{
    "means": dict,        # current distribution mean
    "cov": ndarray,       # current distribution covariance
    "best_theta": dict,   # global best parameters
    "best_loss": float,   # global best loss
    "it": int,            # iteration number
}
```

This enables crash recovery (resume from last completed iteration) and
the external bash loop pattern (Pattern B) where each PBS job runs one
iteration.

## Tuning guidance

| Parameter | Too low | Too high | Rule of thumb |
|-----------|---------|----------|---------------|
| `n_samples` | Poor exploration, misses optima | Expensive per iteration | 10-20× number of free params |
| `n_elite` | Distribution update is noisy | Too conservative, slow convergence | 5-20% of n_samples |
| `max_iter` | May not converge | Wastes compute after convergence | 50 is usually enough; tol handles early stop |
| `tol` | Premature convergence | Never stops | 1e-3 for production, 1e-2 for testing |

### Relationship between n_samples and MPI ranks

Set `n_samples` equal to the number of MPI ranks (= `ncpus` in the PBS
script). Each rank evaluates exactly one candidate per iteration. This
is the most efficient configuration — no rank is idle.

If `n_samples < n_ranks`, some ranks sit idle. If `n_samples > n_ranks`,
some ranks evaluate multiple candidates sequentially.
