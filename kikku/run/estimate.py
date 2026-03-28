"""SMM-style estimation: criterion composition, cross-entropy optimizer, diagnostics."""

from __future__ import annotations

import csv
import gc
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from kikku.run.mpi import bcast_item, is_root, mpi_map

BIG_LOSS = 1e10
NAN_PENALTY = 1e6


def load_estimation_spec(path: str) -> dict[str, Any]:
    """Load estimation YAML and optional precomputed moments CSV.

    Returns
    -------
    dict
        'free', 'moment_spec', 'data_moments', 'method', 'method_options',
        and 'theta_true' if present under ``estimation``.
    """
    p = Path(path).resolve()
    with p.open() as f:
        raw = yaml.safe_load(f)
    est = raw.get("estimation", {}) or {}
    moments = raw.get("moments", {}) or {}
    free = est.get("free", {}) or {}
    method = est.get("method", "cross-entropy")
    method_options = dict(est.get("method_options") or {})
    data_source = (moments.get("data_source") or "precomputed").lower()
    data_moments: dict[str, float] = {}
    if data_source == "precomputed":
        rel = moments.get("data_file", "moments.csv")
        csv_path = Path(rel) if Path(rel).is_absolute() else (p.parent / rel).resolve()
        data_moments = _flatten_moments_csv(csv_path)
    out: dict[str, Any] = {
        "free": free,
        "moment_spec": moments,
        "data_moments": data_moments,
        "method": method,
        "method_options": method_options,
    }
    if "theta_true" in est:
        out["theta_true"] = est["theta_true"]
    return out


def _flatten_moments_csv(csv_path: Path) -> dict[str, float]:
    """Wide CSV: first column is age/group index; remaining columns are moment names."""
    out: dict[str, float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header:
            return out
        names = header[1:]
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                g = int(float(row[0]))
            except ValueError:
                continue
            for j, col in enumerate(names, start=1):
                if j >= len(row):
                    break
                cell = (row[j] or "").strip()
                if cell == "":
                    continue
                try:
                    val = float(cell)
                except ValueError:
                    continue
                key = f"{col}__age{g}"
                out[key] = val
    return out


def _smm_loss(sim_vec: np.ndarray, data_vec: np.ndarray, weights: np.ndarray | None = None) -> float:
    """(sim - data)' W (sim - data) with diagonal W; W = I when weights is None.

    NaN in sim_vec incurs NAN_PENALTY per moment.
    """
    sim_vec = np.asarray(sim_vec, dtype=float)
    data_vec = np.asarray(data_vec, dtype=float)
    diff = sim_vec - data_vec
    nan_mask = np.isnan(sim_vec)
    diff = np.where(nan_mask, 0.0, diff)
    if weights is None:
        loss = float(np.dot(diff, diff))
    else:
        w = np.asarray(weights, dtype=float)
        loss = float(np.dot(diff, w * diff))
    loss += float(np.sum(nan_mask) * NAN_PENALTY)
    return loss


def make_criterion(
    trial_fn: Callable[[dict[str, float]], Any],
    moment_fn: Callable[[Any], dict[str, float]],
    data_moments: dict[str, float],
    weights: np.ndarray | None = None,
) -> Callable[[dict[str, float]], float]:
    """Compose trial + moments + loss into f(theta_dict) -> scalar.

    Aligns dicts by ``sorted(data_moments.keys())``. Missing sim keys become NaN.
    On exception, returns BIG_LOSS. Sets ``criterion.last_sim_moments`` after each
    successful eval for downstream diagnostics.
    """
    # Filter out NaN data moments — they contribute nothing to the loss and
    # propagate NaN through weights and diff, making every loss NaN.
    all_keys = sorted(data_moments.keys())
    keys = [k for k in all_keys if not _is_nan_float(data_moments[k])]
    data_vec = np.array([float(data_moments[k]) for k in keys], dtype=float)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
    else:
        # Default: relative deviations. Weight = 1/data² so the loss is
        # sum((sim-data)²/data²) = sum of squared percentage deviations.
        # For moments with |data| < 1 (correlations, fractions), use absolute
        # deviations (weight=1) to avoid amplifying noise.
        denom = np.where(np.abs(data_vec) >= 1.0, data_vec ** 2, 1.0)
        w_arr = 1.0 / np.maximum(denom, 1e-20)

    def criterion(theta: dict[str, float]) -> float:
        try:
            panels = trial_fn(theta)
            sim_dict = moment_fn(panels)
            del panels  # free simulation arrays eagerly
        except Exception as _exc:
            if not hasattr(criterion, '_err_count'):
                criterion._err_count = 0
            if criterion._err_count < 3:
                import traceback, sys
                print(f"[criterion EXCEPTION] {type(_exc).__name__}: {_exc}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                criterion._err_count += 1
            criterion.last_sim_moments = None
            return BIG_LOSS
        sim_vec = np.array(
            [float(sim_dict[k]) if k in sim_dict and not _is_nan_float(sim_dict[k]) else np.nan for k in keys],
            dtype=float,
        )
        criterion.last_sim_moments = {k: float(sim_dict[k]) for k in sim_dict if not _is_nan_float(sim_dict.get(k))}
        return _smm_loss(sim_vec, data_vec, w_arr)

    criterion.last_sim_moments = None  # type: ignore[attr-defined]
    return criterion


def _is_nan_float(x: Any) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return True


@dataclass
class EstimationResult:
    theta: dict[str, float]
    objective: float
    converged: bool
    n_iter: int
    history: list[dict[str, Any]] = field(default_factory=list)
    sim_moments: dict[str, float] = field(default_factory=dict)


def estimate(
    criterion: Callable[[dict[str, float]], float],
    param_spec: dict[str, Any],
    method: str = "cross-entropy",
    method_options: dict[str, Any] | None = None,
    comm: Any = None,
    verbose: bool = True,
) -> EstimationResult:
    """Minimize criterion via cross-entropy method."""
    opts = dict(method_options or {})
    if method in ("cross-entropy", "ce", "cross_entropy"):
        return _cross_entropy_minimize(criterion, param_spec, opts, comm, verbose)
    # TODO: add alternative methods when needed
    raise ValueError(f"unknown method: {method!r}")


_SAFE_CRITERION_LOG_COUNT = 0

def _safe_criterion(criterion: Callable[[dict[str, float]], float], theta: dict[str, float]) -> float:
    global _SAFE_CRITERION_LOG_COUNT
    try:
        return float(criterion(theta))
    except Exception as _exc:
        if _SAFE_CRITERION_LOG_COUNT < 3:
            import traceback, sys
            print(f"[criterion EXCEPTION] {type(_exc).__name__}: {_exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            _SAFE_CRITERION_LOG_COUNT += 1
        return BIG_LOSS


def _param_vector(theta: dict[str, float], names: list[str]) -> np.ndarray:
    return np.array([float(theta[n]) for n in names], dtype=float)


def _dict_from_vector(x: np.ndarray, names: list[str]) -> dict[str, float]:
    return {names[i]: float(x[i]) for i in range(len(names))}


def _sample_uniform(
    param_spec: dict[str, Any],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Draw uniformly within bounds (iteration 0 of CE, Eggsandbaskets/fempres pattern)."""
    names = sorted(param_spec.keys())
    out: list[dict[str, float]] = []
    for _ in range(n_samples):
        theta = {}
        for n in names:
            lo, hi = float(param_spec[n]["bounds"][0]), float(param_spec[n]["bounds"][1])
            theta[n] = float(rng.uniform(lo, hi))
        out.append(theta)
    return out


def _sample_bounded(
    means: dict[str, float],
    cov: np.ndarray,
    param_spec: dict[str, Any],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    names = sorted(param_spec.keys())
    mean_vec = np.array([float(means[n]) for n in names], dtype=float)
    cov_m = np.asarray(cov, dtype=float)
    if cov_m.ndim == 1:
        cov_m = np.diag(cov_m)
    out: list[dict[str, float]] = []
    for _ in range(n_samples):
        accepted = False
        z = mean_vec
        for _retry in range(1000):
            z = rng.multivariate_normal(mean_vec, cov_m)
            theta = _dict_from_vector(z, names)
            if all(
                float(param_spec[n]["bounds"][0]) <= theta[n] <= float(param_spec[n]["bounds"][1])
                for n in names
            ):
                out.append(theta)
                accepted = True
                break
        if not accepted:
            clipped = np.array(
                [
                    np.clip(
                        z[i],
                        float(param_spec[names[i]]["bounds"][0]),
                        float(param_spec[names[i]]["bounds"][1]),
                    )
                    for i in range(len(names))
                ],
                dtype=float,
            )
            out.append(_dict_from_vector(clipped, names))
    return out


def _elite_weighted_mean_cov(
    thetas: list[dict[str, float]], losses: list[float], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Exp weights (fempres-style): w_i ∝ exp(-(L_i - min L))."""
    L = np.asarray(losses, dtype=float)
    exw = np.exp(-(L - np.min(L)))
    s = float(np.sum(exw))
    if s <= 0 or not np.isfinite(s):
        exw = np.ones(len(L), dtype=float) / max(len(L), 1)
    else:
        exw = exw / s
    X = np.stack([_param_vector(t, names) for t in thetas], axis=0)
    mean_vec = exw @ X
    xc = X - mean_vec
    cov = (xc.T * exw) @ xc
    eps = 1e-12
    cov = cov + np.eye(len(names)) * eps
    return mean_vec, cov


def _cross_entropy_minimize(
    criterion: Callable[[dict[str, float]], float],
    param_spec: dict[str, Any],
    options: dict[str, Any],
    comm: Any,
    verbose: bool,
) -> EstimationResult:
    names = sorted(param_spec.keys())
    n_samples = int(options.get("n_samples", 48))
    n_elite = int(options.get("n_elite", 10))
    max_iter = int(options.get("max_iter", 50))
    tol = float(options.get("tol", 1e-2))
    # Two independent seeds:
    #   sampling_seed  — RNG for CE parameter draws (which thetas to explore)
    #   simulation_seed — passed to trial function for CRN (same shocks for all thetas)
    # Falls back to legacy 'seed' key for backward compat.
    sampling_seed = int(options.get("sampling_seed", options.get("seed", 0)))
    simulation_seed = int(options.get("simulation_seed", 99))
    noise_fraction = float(options.get("noise_fraction", 0.0))
    checkpoint_dir = options.get("checkpoint_dir")

    n_elite = max(1, min(n_elite, n_samples))

    # Initialise: no means/cov yet — iteration 0 draws uniform
    means: dict[str, float] | None = None
    cov: np.ndarray | None = None

    rng = np.random.default_rng(sampling_seed)
    history: list[dict[str, Any]] = []
    best_theta: dict[str, float] = {
        n: 0.5 * (float(param_spec[n]["bounds"][0]) + float(param_spec[n]["bounds"][1]))
        for n in names
    }
    best_loss = BIG_LOSS
    converged = False
    elite_mean_loss_prev: float | None = None
    rss_post_gc_prev = 0

    for it in range(max_iter):
        # Iteration 0: uniform over bounds (Eggsandbaskets/fempres pattern)
        # Iteration 1+: truncated MVN from elite distribution, with
        # noise_fraction of draws replaced by uniform samples to maintain
        # exploration and prevent premature collapse.
        if is_root(comm):
            if it == 0 or means is None:
                candidates = _sample_uniform(param_spec, n_samples, rng)
            else:
                n_noise = max(0, int(round(noise_fraction * n_samples)))
                n_elite_draws = n_samples - n_noise
                candidates = _sample_bounded(means, cov, param_spec, n_elite_draws, rng)
                if n_noise > 0:
                    candidates.extend(_sample_uniform(param_spec, n_noise, rng))
        else:
            candidates = None
        candidates = bcast_item(candidates, comm, root=0)
        assert candidates is not None

        losses = mpi_map(lambda th: _safe_criterion(criterion, th), candidates, comm=comm)

        if is_root(comm):
            assert losses is not None
            paired = sorted(zip(candidates, losses), key=lambda p: p[1])
            elite_pairs = paired[:n_elite]
            elite_thetas = [p[0] for p in elite_pairs]
            elite_losses = [float(p[1]) for p in elite_pairs]

            for th, ell in zip(elite_thetas, elite_losses):
                if ell < best_loss:
                    best_loss = ell
                    best_theta = dict(th)

            mean_vec, cov = _elite_weighted_mean_cov(elite_thetas, elite_losses, names)
            means = _dict_from_vector(mean_vec, names)

            elite_mean_loss = float(np.mean(elite_losses))
            history.append(
                {
                    "means": dict(means),
                    "cov": np.array(cov, copy=True),
                    "best_loss": best_loss,
                    "elite_mean_loss": elite_mean_loss,
                }
            )

            if checkpoint_dir:
                cdir = Path(checkpoint_dir)
                cdir.mkdir(parents=True, exist_ok=True)
                state = {"means": means, "cov": cov, "best_theta": best_theta, "best_loss": best_loss, "it": it}
                with (cdir / "state.pkl").open("wb") as f:
                    pickle.dump(state, f)

            if elite_mean_loss_prev is not None:
                if abs(elite_mean_loss_prev - elite_mean_loss) < tol:
                    converged = True
            elite_mean_loss_prev = elite_mean_loss

            if verbose:
                theta_str = '  '.join(f'{n}={best_theta[n]:.4f}' for n in names)
                print(f"[ce] iter={it + 1}/{max_iter} best_loss={best_loss:.6f} "
                      f"elite_mean={elite_mean_loss:.6f}  {theta_str}")

        # --- Memory diagnostics (rank 0, every iteration) ---
        if is_root(comm) and verbose:
            try:
                import resource
                rss_post_eval = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            except Exception:
                rss_post_eval = 0

        converged = bcast_item(converged if is_root(comm) else None, comm, root=0)
        means = bcast_item(means if is_root(comm) else None, comm, root=0)
        cov = bcast_item(cov if is_root(comm) else None, comm, root=0)
        best_theta = bcast_item(best_theta if is_root(comm) else None, comm, root=0)
        best_loss = bcast_item(best_loss if is_root(comm) else None, comm, root=0)
        history = bcast_item(history if is_root(comm) else None, comm, root=0)

        if is_root(comm) and verbose:
            try:
                rss_post_bcast = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            except Exception:
                rss_post_bcast = 0

        if converged:
            break

        # Free any solution arrays lingering from this iteration's evals
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (OSError, AttributeError):
            pass

        if is_root(comm) and verbose:
            try:
                rss_post_gc = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
                print(f"  [mem] iter={it+1} eval={rss_post_eval}MB "
                      f"bcast={rss_post_bcast}MB gc={rss_post_gc}MB "
                      f"delta_eval={rss_post_eval - (rss_post_gc_prev if it > 0 else rss_post_eval)}MB "
                      f"delta_bcast={rss_post_bcast - rss_post_eval}MB "
                      f"delta_gc={rss_post_gc - rss_post_bcast}MB")
                rss_post_gc_prev = rss_post_gc
            except Exception:
                pass

    # Use last_sim_moments from the CE loop if available, avoiding a
    # costly re-evaluation that can OOM on large grids.
    sim_moments: dict[str, float] = {}
    if is_root(comm):
        cached = getattr(criterion, "last_sim_moments", None)
        if cached:
            sim_moments = dict(cached)
        else:
            try:
                _safe_criterion(criterion, best_theta)
                sim_moments = dict(getattr(criterion, "last_sim_moments", None) or {})
            except Exception:
                sim_moments = {}
    sim_moments = bcast_item(sim_moments if is_root(comm) else None, comm, root=0)

    return EstimationResult(
        theta=dict(best_theta),
        objective=float(best_loss),
        converged=converged,
        n_iter=len(history),
        history=list(history),
        sim_moments=dict(sim_moments or {}),
    )


def diagnostics(
    result: EstimationResult,
    data_moments: dict[str, float],
    moment_names: dict[str, str] | list[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit table, total loss, and top contributors."""
    keys = sorted(data_moments.keys())
    sim = result.sim_moments or {}
    w = None if weights is None else np.asarray(weights, dtype=float)

    name_map: dict[str, str]
    if moment_names is None:
        name_map = {k: k for k in keys}
    elif isinstance(moment_names, list):
        name_map = {k: (moment_names[i] if i < len(moment_names) else k) for i, k in enumerate(keys)}
    else:
        name_map = {k: moment_names.get(k, k) for k in keys}

    fit_table: list[dict[str, Any]] = []
    contrib_list: list[tuple[str, float]] = []

    for i, k in enumerate(keys):
        d = float(data_moments[k])
        s = float(sim[k]) if k in sim and not _is_nan_float(sim.get(k)) else np.nan
        resid = float(s - d) if np.isfinite(s) else np.nan
        if not np.isfinite(s):
            raw = NAN_PENALTY
        else:
            diff = s - d
            if w is None:
                raw = float(diff * diff)
            else:
                wi = float(w[i]) if i < len(w) else 1.0
                raw = float(wi * diff * diff)
        contrib_list.append((k, raw))
        fit_table.append(
            {
                "moment": name_map.get(k, k),
                "data": d,
                "simulated": s,
                "residual": resid,
                "contribution": raw,
            }
        )

    total_loss = float(result.objective)
    denom = sum(c for _, c in contrib_list)
    if denom > 0 and np.isfinite(denom):
        for row in fit_table:
            row["contribution_pct"] = 100.0 * float(row["contribution"]) / denom
    else:
        for row in fit_table:
            row["contribution_pct"] = 0.0

    contrib_sorted = sorted(contrib_list, key=lambda t: t[1], reverse=True)
    worst_moments = [
        {
            "moment": name_map.get(k, k),
            "contribution": c,
            "data": float(data_moments[k]),
            "simulated": float(sim[k]) if k in sim and not _is_nan_float(sim.get(k)) else float("nan"),
        }
        for k, c in contrib_sorted[:5]
    ]

    return {
        "fit_table": fit_table,
        "total_loss": total_loss,
        "worst_moments": worst_moments,
    }
