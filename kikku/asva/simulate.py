"""Generic forward Monte Carlo simulator for DDSL models.

Walks the period graph (from ``kikku.dynx.graphs``) in
``forward_order``, applying transition callables and
evaluating solved policies at each stage.  Model-specific
parts are injected as callables; the loop is model-agnostic.

Mathematical object: the **pushforward measure operator**,
conjugate to the backward Bellman operator.

Shocks are **pre-drawn** before the simulation loop for
reproducibility and JIT compatibility.  Each model provides
a ``draw_shocks`` function that returns a ``draws`` dict
(keyed by ``(t, point)``) of ``(N,)`` arrays.  The walker
indexes into this dict instead of consuming an rng inline.

Shock injection points (per-transition, not per-stage):

    =========== =================== ===========================
    Point       When                Backward conjugate
    =========== =================== ===========================
    shock_arvl  before agent acts   E in dcsn_to_arvl_mover
    shock_cntn  after agent acts    E inside cntn_to_dcsn_mover
    twister     between periods     inter-period expectation
    =========== =================== ===========================
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from kikku.dynx.graphs import forward_order


# ── Types ──────────────────────────────────────────────────

StageForward = namedtuple("StageForward", [
    "arvl_to_dcsn",       # (particles, shocks) → particles
    "policy",             # (particles) → controls
    "dcsn_to_cntn",       # (particles, controls, shocks) → particles
    "shock_draw_arvl",    # str key into draws dict | None
    "shock_draw_cntn",    # str key into draws dict | None
])

BranchingForward = namedtuple("BranchingForward", [
    "arvl_to_dcsn",       # (particles, shocks) → particles
    "branch_policy",      # (particles) → labels array
    "dcsn_to_cntn",       # dict[label → callable(particles, shocks) → particles]
    "shock_draw_arvl",    # str key into draws dict | None
    "shock_draw_cntn",    # str key into draws dict | None
])


# ── Particle helpers ───────────────────────────────────────

def _slice(particles: dict, mask: np.ndarray) -> dict:
    """Select agents matching boolean mask."""
    return {k: v[mask] for k, v in particles.items()}


def _scatter(target: dict, source: dict, mask: np.ndarray):
    """Write *source* values back into *target* at *mask* positions."""
    for k, v in source.items():
        if k not in target:
            if np.issubdtype(np.asarray(v).dtype, np.floating):
                target[k] = np.full(mask.shape[0], np.nan)
            else:
                target[k] = np.zeros(mask.shape[0], dtype=np.asarray(v).dtype)
        target[k][mask] = v


def _scatter_rec(records: dict, stage_id: str,
                 branch_rec: dict, mask: np.ndarray, N: int):
    """Write branch records into full-population records dict."""
    if stage_id not in records:
        records[stage_id] = {}
    for key, val in branch_rec.items():
        if isinstance(val, dict):
            if key not in records[stage_id]:
                records[stage_id][key] = {}
            for k2, v2 in val.items():
                if k2 not in records[stage_id][key]:
                    arr = np.asarray(v2)
                    if np.issubdtype(arr.dtype, np.floating):
                        records[stage_id][key][k2] = np.full(N, np.nan)
                    else:
                        records[stage_id][key][k2] = np.zeros(N, dtype=arr.dtype)
                records[stage_id][key][k2][mask] = v2
        else:
            if key not in records[stage_id]:
                arr = np.asarray(val)
                if np.issubdtype(arr.dtype, np.floating):
                    records[stage_id][key] = np.full(N, np.nan)
                else:
                    records[stage_id][key] = np.zeros(N, dtype=arr.dtype)
            records[stage_id][key][mask] = val


# ── Rename (connectors / twisters) ────────────────────────

def apply_rename(rename: dict, particles: dict) -> dict:
    """Apply a connector or twister rename map."""
    if not rename:
        return particles
    return {rename.get(k, k): v for k, v in particles.items()}


# ── Single-stage forward step ─────────────────────────────

def forward_stage(sfwd: StageForward, particles: dict,
                  draws_t: dict) -> tuple[dict, dict]:
    """Walk one standard stage: arvl → dcsn → cntn.

    Parameters
    ----------
    sfwd : StageForward
    particles : dict[str, ndarray]
    draws_t : dict
        Pre-drawn shocks for this period, keyed by the
        shock point names declared in the StageForward.

    Returns (poststates, records).
    """
    shocks_a = draws_t.get(sfwd.shock_draw_arvl, {}) if sfwd.shock_draw_arvl else {}
    states = sfwd.arvl_to_dcsn(particles, shocks_a)
    controls = sfwd.policy(states)
    shocks_c = draws_t.get(sfwd.shock_draw_cntn, {}) if sfwd.shock_draw_cntn else {}
    poststates = sfwd.dcsn_to_cntn(states, controls, shocks_c)
    return poststates, {"states": states, "controls": controls,
                        "poststates": poststates}


# ── Period-level graph walker ─────────────────────────────

def simulate_period(graph, stage_forwards: dict,
                    particles: dict,
                    draws_t: dict) -> tuple[dict, dict]:
    """Walk one period forward through the stage DAG.

    Parameters
    ----------
    graph : nx.DiGraph
        Period graph from ``period_to_graph``.
    stage_forwards : dict[str, StageForward | BranchingForward]
        Compiled forward callables per stage.
    particles : dict[str, ndarray]
        Agent states ``{field: array[N]}``.
    draws_t : dict
        Pre-drawn shocks for this period.  Keyed by shock
        point names (strings).  Each value is an ndarray
        of shape ``(N,)`` or a dict of such arrays.

    Returns
    -------
    particles : dict[str, ndarray]
    records : dict[str, dict]
    """
    records: Dict[str, dict] = {}
    visited: set = set()
    N = len(next(iter(particles.values())))

    for stage_id in forward_order(graph):
        if stage_id in visited:
            continue

        node = graph.nodes[stage_id]
        sfwd = stage_forwards[stage_id]

        if node["kind"] == "branching":
            shocks_a = draws_t.get(sfwd.shock_draw_arvl, {}) if sfwd.shock_draw_arvl else {}
            states = sfwd.arvl_to_dcsn(particles, shocks_a)
            labels = sfwd.branch_policy(states)
            records[stage_id] = {"states": states, "labels": labels}

            for successor in graph.successors(stage_id):
                edge = graph.edges[stage_id, successor]
                branch_label = edge.get("branch")
                rename = edge.get("rename", {})
                visited.add(successor)

                mask = (labels == branch_label)
                if not np.any(mask):
                    continue

                branch_p = _slice(states, mask)

                shocks_c = draws_t.get(sfwd.shock_draw_cntn, {}) if sfwd.shock_draw_cntn else {}
                if isinstance(shocks_c, dict):
                    shocks_c = {k: v[mask] if isinstance(v, np.ndarray) else v
                                for k, v in shocks_c.items()}
                branch_p = sfwd.dcsn_to_cntn[branch_label](branch_p, shocks_c)
                branch_p = apply_rename(rename, branch_p)

                succ_fwd = stage_forwards[successor]
                bp, br = forward_stage(succ_fwd, branch_p, draws_t)

                _scatter_rec(records, successor, br, mask, N)
                _scatter(particles, bp, mask)

        else:
            particles, rec = forward_stage(sfwd, particles, draws_t)
            records[stage_id] = rec

            for successor in graph.successors(stage_id):
                edge = graph.edges[stage_id, successor]
                rename = edge.get("rename", {})
                particles = apply_rename(rename, particles)

    return particles, records


# ── Shock pre-drawing ─────────────────────────────────────

def draw_shocks(N: int, t0: int, T_end: int,
                shock_spec: dict, rng_seed: int = 42) -> dict:
    """Pre-draw all Monte Carlo shocks before the simulation.

    Parameters
    ----------
    N : int
        Number of agents.
    t0, T_end : int
        Period range (inclusive).
    shock_spec : dict
        Declares which shocks to draw and how.  Keyed by
        shock point name (the same string used in
        ``StageForward.shock_draw_arvl`` etc.).  Each value
        is a callable ``(N, rng) → ndarray(N,)`` or dict
        of arrays.
    rng_seed : int

    Returns
    -------
    draws : dict[int, dict]
        ``draws[t][point_name]`` = pre-drawn shock array
        for period *t* and shock point *point_name*.
    """
    rng = np.random.default_rng(rng_seed)
    draws = {}
    for t in range(t0, T_end + 1):
        draws[t] = {}
        for name, draw_fn in shock_spec.items():
            draws[t][name] = draw_fn(N, rng)
    return draws


# ── Top-level simulate ────────────────────────────────────

def simulate(
    graph,
    twister: dict,
    stage_forwards_by_t: Dict[int, dict],
    twister_fn: Optional[Callable] = None,
    initial_particles: Optional[dict] = None,
    t0: int = 0,
    T_end: int = 19,
    draws: Optional[dict] = None,
    rng_seed: int = 42,
) -> tuple[dict, dict]:
    """Generic forward Monte Carlo simulator.

    Parameters
    ----------
    graph : nx.DiGraph
        Period graph (topology is the same every period).
    twister : dict
        Inter-period rename map from ``nest.yaml``.
    stage_forwards_by_t : dict[int, dict[str, StageForward]]
        Per-period compiled callables.
    twister_fn : callable, optional
        ``(particles, draws_t) → shocks_dict``.
        Post-twister shock injection (e.g. Markov draw).
        Receives pre-drawn shocks from ``draws[t]``.
    initial_particles : dict[str, ndarray]
    t0, T_end : int
        Calendar age range (inclusive).
    draws : dict[int, dict], optional
        Pre-drawn shocks from :func:`draw_shocks`.
        If None, no shocks are drawn (deterministic model).
    rng_seed : int
        Used only if ``draws`` is None and model has no shocks.

    Returns
    -------
    history : dict[int, dict]
    final_particles : dict[str, ndarray]
    """
    particles = {k: v.copy() for k, v in initial_particles.items()}
    history: Dict[int, dict] = {}
    if draws is None:
        draws = {t: {} for t in range(t0, T_end + 1)}

    for t in range(t0, T_end + 1):
        draws_t = draws.get(t, {})
        stage_forwards = stage_forwards_by_t[t]
        particles, records = simulate_period(
            graph, stage_forwards, particles, draws_t)
        records["t"] = t
        history[t] = records

        if t < T_end:
            particles = apply_rename(twister, particles)
            if twister_fn is not None:
                draws_next = draws.get(t + 1, {})
                shocks = twister_fn(particles, draws_next)
                particles.update(shocks)

    return history, particles
