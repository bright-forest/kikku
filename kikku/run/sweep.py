"""Parameter sweep infrastructure.

Runs a solve function across a parameter grid, collects metrics,
handles best-of-n timing, and optionally distributes via MPI.
"""

import itertools
import time
import warnings
import math
from typing import Callable, Any

from .mpi import scatter_items, gather_results, is_root


def param_grid(**axes) -> list:
    """Cartesian product of parameter axes.

    >>> param_grid(grid_size=[500, 1000], delta=[0.5, 1.0])
    [{'grid_size': 500, 'delta': 0.5},
     {'grid_size': 500, 'delta': 1.0},
     {'grid_size': 1000, 'delta': 0.5},
     {'grid_size': 1000, 'delta': 1.0}]
    """
    keys = list(axes.keys())
    vals = list(axes.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def sweep(
    solve_fn: Callable[[dict], Any],
    param_grid: list,
    metric_fns: dict,
    n_reps: int = 3,
    warmup: bool = True,
    best: str = 'min',
    on_error: str = 'raise',
    verbose: bool = True,
    comm=None,
) -> list:
    """Run solve_fn across param_grid, best-of-n, collect metrics.

    Parameters
    ----------
    solve_fn :
        (overrides_dict) -> result.  Result is opaque; metric_fns
        know how to extract numbers from it.
    param_grid :
        Each dict is a parameter combination.
    metric_fns :
        {name: result -> float}.  First key is the primary metric
        used for best-of-n selection.
    n_reps :
        Number of repetitions per parameter combination (best of n).
    warmup :
        If True, run solve_fn once before timing. Result is discarded
        for metric selection.
    best :
        'min' or 'max' — whether best = min or max of the primary
        metric (first key in metric_fns).
    on_error :
        'raise' (default): propagate exceptions.
        'skip': log warning, continue, mark failed points with NaN.
    verbose :
        Print progress.
    comm :
        MPI communicator (None for single-process).

    Returns
    -------
    list[dict]
        Each dict is flat: {**overrides, **metrics}.
    """
    local_points = scatter_items(param_grid, comm)
    n_total = len(param_grid)
    primary_key = next(iter(metric_fns))
    compare = min if best == 'min' else max

    results = []
    for idx, ov in enumerate(local_points):
        if verbose and is_root(comm):
            print(f"Point {idx + 1}/{len(local_points)} "
                  f"(of {n_total} total): {ov}")

        # Warmup (discard result for selection)
        if warmup:
            try:
                solve_fn(ov)
            except Exception as e:
                if on_error == 'raise':
                    raise
                warnings.warn(f"Warmup failed for {ov}: {e}")

        best_metrics = None
        best_primary = None

        for rep in range(n_reps):
            if verbose and is_root(comm):
                print(f"  rep {rep + 1}/{n_reps}", end="", flush=True)

            try:
                result = solve_fn(ov)
                metrics = {
                    name: fn(result) for name, fn in metric_fns.items()
                }
            except Exception as e:
                if on_error == 'raise':
                    raise
                warnings.warn(f"Failed {ov} rep {rep}: {e}")
                metrics = {name: math.nan for name in metric_fns}

            primary_val = metrics[primary_key]

            if best_primary is None or (
                not math.isnan(primary_val) and
                compare(primary_val, best_primary) == primary_val
            ):
                best_primary = primary_val
                best_metrics = metrics

            if verbose and is_root(comm):
                print(f"  {primary_key}={primary_val:.4g}")

        if best_metrics is None:
            best_metrics = {name: math.nan for name in metric_fns}

        results.append({**ov, **best_metrics})

    return gather_results(results, comm)


