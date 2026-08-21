"""Parameter sweep: MAP a solve function over a test set (v2, §7.4)."""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from .mpi import gather_results, is_root, scatter_items

P = TypeVar("P")


@dataclass(frozen=True)
class SweepResult:
    """One row: point (e.g. ``TestSpec``) + best-of-``n_reps`` metrics (§7.4)."""

    point: Any
    metrics: dict[str, Any]
    result: Any | None = None


def param_grid(**axes) -> list:
    """Cartesian product of **named** scalar axes (legacy helper, not v2 make_test)."""
    keys = list(axes.keys())
    vals = list(axes.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def sweep(
    solve_fn: Callable[[P], Any],
    points: Sequence[P],
    metric_fns: dict,
    n_reps: int = 3,
    warmup: bool = True,
    best: str = "min",
    on_error: str = "raise",
    verbose: bool = True,
    comm=None,
) -> list[SweepResult]:
    """``MAP`` ``solve_fn`` over ``points``; best-of-``n_reps``; optional MPI (§7.4)."""
    local_points = list(scatter_items(list(points), comm))
    n_total = len(points)
    primary_key = next(iter(metric_fns))
    compare = min if best == "min" else max

    results: list[SweepResult] = []
    for idx, point in enumerate(local_points):
        if verbose and is_root(comm):
            print(
                f"Point {idx + 1}/{len(local_points)} (of {n_total} total): {point}"
            )
        if warmup:
            try:
                solve_fn(point)
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                warnings.warn(f"Warmup failed for {point!r}: {e}")

        best_metrics: dict[str, float] | None = None
        best_primary: float | None = None
        best_res: Any = None
        for rep in range(n_reps):
            if verbose and is_root(comm):
                print(f"  rep {rep + 1}/{n_reps}", end="", flush=True)
            try:
                res = solve_fn(point)
                metrics = {name: fn(res) for name, fn in metric_fns.items()}
            except Exception as e:  # noqa: BLE001
                if on_error == "raise":
                    raise
                warnings.warn(f"Failed {point!r} rep {rep}: {e}")
                metrics = {name: math.nan for name in metric_fns}
                res = None
            primary_val = metrics[primary_key]
            if best_primary is None or (
                not math.isnan(primary_val)
                and compare(primary_val, best_primary) == primary_val
            ):
                best_primary = primary_val
                best_metrics = dict(metrics)
                if res is not None:
                    best_res = res
            if verbose and is_root(comm):
                print(f"  {primary_key}={primary_val:.4g}")
        if best_metrics is None:
            best_metrics = {name: math.nan for name in metric_fns}
        results.append(
            SweepResult(point=point, metrics=best_metrics, result=best_res)
        )
    return gather_results(results, comm)
