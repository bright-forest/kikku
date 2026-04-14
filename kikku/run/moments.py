"""Moment engine: panels dict or DataFrame -> dict[str, float].

Only depends on numpy and pandas. No estimation core or model imports.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

import numpy as np
import pandas as pd

MIN_OBS = 10

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_NP_ALLOWED = frozenset({"log", "maximum", "minimum", "abs"})


# ---------------------------------------------------------------------------
# Statistical aggregators
# ---------------------------------------------------------------------------
# (_mean, _sd, _corr, _autocorr, _fraction)

def _mean(arr: np.ndarray) -> float:
    if np.sum(~np.isnan(arr)) < MIN_OBS:
        return float("nan")
    return float(np.nanmean(arr))


def _sd(arr: np.ndarray) -> float:
    if np.sum(~np.isnan(arr)) < MIN_OBS:
        return float("nan")
    return float(np.nanstd(arr, ddof=0))


def _corr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    mask = ~(np.isnan(arr1) | np.isnan(arr2))
    if np.sum(mask) < MIN_OBS:
        return float("nan")
    c = np.corrcoef(arr1[mask], arr2[mask])[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _autocorr(panel_var: np.ndarray, t_indices: list[int]) -> float:
    T = panel_var.shape[0]
    x_t_list: list[np.ndarray] = []
    x_tp1_list: list[np.ndarray] = []
    for t in t_indices:
        if t + 1 < T:
            x_t_list.append(panel_var[t])
            x_tp1_list.append(panel_var[t + 1])
    if not x_t_list:
        return float("nan")
    x_t = np.concatenate(x_t_list)
    x_tp1 = np.concatenate(x_tp1_list)
    return _corr(x_t, x_tp1)


def _fraction(arr: np.ndarray, value: Any) -> float:
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_OBS:
        return float("nan")
    return float(np.mean(valid == value))


def _age_group_masks(
    T: int,
    age_groups: dict,
    t0: int = 0,
) -> dict[str, list[int]]:
    masks: dict[str, list[int]] = {}
    for name, bounds in age_groups.items():
        age_lo, age_hi = bounds[0], bounds[1]
        t_lo = int(age_lo - t0)
        t_hi = int(age_hi - t0)
        indices = [t for t in range(max(0, t_lo), min(T, t_hi))]
        masks[str(name)] = indices
    return masks


# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------
# (_safe_eval_expr, _safe_eval, _names_in_expr)

def _safe_eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)) or node.value is None:
            return node.value
        raise ValueError(f"disallowed constant type: {type(node.value)}")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return _safe_eval_expr(node.operand, env)
        raise ValueError("disallowed unary op")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOPS:
            raise ValueError("disallowed binary op")
        left = _safe_eval_expr(node.left, env)
        right = _safe_eval_expr(node.right, env)
        return _BINOPS[op_type](left, right)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Attribute):
            raise ValueError("only np.* calls allowed")
        if not isinstance(func.value, ast.Name) or func.value.id != "np":
            raise ValueError("only np.* calls allowed")
        if func.attr not in _NP_ALLOWED:
            raise ValueError(f"np.{func.attr} not allowed")
        fn = getattr(np, func.attr)
        args = [_safe_eval_expr(a, env) for a in node.args]
        return fn(*args)
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def _safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval_expr(tree, env)


def _names_in_expr(expr: str) -> set[str]:
    tree = ast.parse(expr.strip(), mode="eval")
    names: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            if n.id != "np":
                names.add(n.id)

    V().visit(tree)
    return names


# ---------------------------------------------------------------------------
# Derived variables
# ---------------------------------------------------------------------------

def _add_derived_variables(
    panels: dict[str, np.ndarray],
    derived_spec: dict[str, str],
) -> None:
    env = {name: arr for name, arr in panels.items()}
    env["np"] = np
    for var_name, expr in derived_spec.items():
        panels[var_name] = _safe_eval(expr, env)
        env[var_name] = panels[var_name]


def _collect_variable_names(spec: dict) -> list[str]:
    names: set[str] = set()
    ident = spec.get("identification") or {}
    for var in ident.get("mean", []) or []:
        names.add(str(var))
    for var in (ident.get("sd", []) or ident.get("sds", []) or []):
        names.add(str(var))
    for pair in ident.get("corrs", []) or []:
        names.add(str(pair[0]))
        names.add(str(pair[1]))
    for var in ident.get("autocorrs", []) or []:
        names.add(str(var))
    for var in ident.get("branch_fractions", []) or []:
        names.add(str(var))
    cond = ident.get("conditional") or {}
    for cond_var, branches in cond.items():
        names.add(str(cond_var))
        if not isinstance(branches, dict):
            continue
        for _, branch_moments in branches.items():
            if not isinstance(branch_moments, dict):
                continue
            for var in branch_moments.get("mean", []) or []:
                names.add(str(var))
            for var in branch_moments.get("sd", []) or []:
                names.add(str(var))
            for pair in branch_moments.get("corrs", []) or []:
                names.add(str(pair[0]))
                names.add(str(pair[1]))
    for target in spec.get("targets", []) or []:
        if "var" in target:
            names.add(str(target["var"]))
        if "vars" in target:
            for v in target["vars"]:
                names.add(str(v))
    dv = spec.get("derived_variables")
    if dv is None and spec.get("setup"):
        dv = (spec.get("setup") or {}).get("derived_variables")
    if isinstance(dv, dict):
        for expr in dv.values():
            names |= _names_in_expr(str(expr))
    return sorted(names)


# ---------------------------------------------------------------------------
# Core moment computation
# ---------------------------------------------------------------------------

def _make_pool_fn(
    panels: dict[str, np.ndarray],
    T: int,
    t_indices: list[int],
):
    def _pool(var_name: str) -> np.ndarray:
        arr = panels[var_name]
        rows = [arr[t] for t in t_indices if t < T]
        if not rows:
            return np.array([])
        return np.stack(rows, axis=0).ravel()

    return _pool


def compute_moments_from_panels(panels: dict, spec: dict) -> dict[str, float]:
    """Core: {var: (T,N) array} -> dict[str, float]."""
    panels = dict(panels)
    first = next(iter(panels.values()))
    T, N = int(first.shape[0]), int(first.shape[1])

    dv = spec.get("derived_variables")
    if dv is None:
        dv = (spec.get("setup") or {}).get("derived_variables")
    if isinstance(dv, dict) and dv:
        _add_derived_variables(panels, dv)

    setup = spec.get("setup") or {}
    t0 = int(setup.get("t0", 0))
    age_groups = setup.get("age_groups") or {}
    if not age_groups:
        age_groups = {"all": [t0, t0 + T]}
    masks = _age_group_masks(T, age_groups, t0)

    ident = spec.get("identification") or {}
    result: dict[str, float] = {}

    for ag_name, t_indices in masks.items():
        if not t_indices:
            continue
        _pool = _make_pool_fn(panels, T, t_indices)

        for var in ident.get("mean", []) or []:
            vs = str(var)
            result[f"mean_{vs}__age{ag_name}"] = _mean(_pool(vs))

        sd_vars = ident.get("sd", []) or ident.get("sds", []) or []
        for var in sd_vars:
            vs = str(var)
            result[f"sd_{vs}__age{ag_name}"] = _sd(_pool(vs))

        for var_pair in ident.get("corrs", []) or []:
            v1, v2 = str(var_pair[0]), str(var_pair[1])
            result[f"corr_{v1}_{v2}__age{ag_name}"] = _corr(
                _pool(v1), _pool(v2)
            )

        for var in ident.get("autocorrs", []) or []:
            vs = str(var)
            result[f"autocorr_{vs}__age{ag_name}"] = _autocorr(
                panels[vs], t_indices
            )

        for var in ident.get("branch_fractions", []) or []:
            vs = str(var)
            raw = panels[vs]
            flat = raw[~np.isnan(raw)]
            if flat.size == 0:
                continue
            unique_vals = np.unique(flat)
            for val in unique_vals:
                if not np.isfinite(val):
                    continue
                val_int = int(val)
                result[f"branch_frac_{vs}_{val_int}__age{ag_name}"] = _fraction(
                    _pool(vs), val_int
                )

        cond_spec = ident.get("conditional") or {}
        for cond_var, branches in cond_spec.items():
            cv = str(cond_var)
            if not isinstance(branches, dict):
                continue
            cond_pool = _pool(cv)
            for branch_val, branch_moments in branches.items():
                branch_val_int = int(branch_val)
                if not isinstance(branch_moments, dict):
                    continue

                def _cond_pool(var_name: str) -> np.ndarray:
                    pool = _pool(str(var_name))
                    mask = cond_pool == float(branch_val_int)
                    out = np.full_like(pool, np.nan, dtype=float)
                    out[mask] = pool[mask]
                    return out

                for var in branch_moments.get("mean", []) or []:
                    vs = str(var)
                    key = (
                        f"cond_{cv}_{branch_val_int}_mean_{vs}__age{ag_name}"
                    )
                    result[key] = _mean(_cond_pool(vs))

                for var in branch_moments.get("sd", []) or []:
                    vs = str(var)
                    key = f"cond_{cv}_{branch_val_int}_sd_{vs}__age{ag_name}"
                    result[key] = _sd(_cond_pool(vs))

                for var_pair in branch_moments.get("corrs", []) or []:
                    v1, v2 = str(var_pair[0]), str(var_pair[1])
                    key = (
                        f"cond_{cv}_{branch_val_int}_corr_{v1}_{v2}__age{ag_name}"
                    )
                    result[key] = _corr(_cond_pool(v1), _cond_pool(v2))

    for target in spec.get("targets", []) or []:
        key_base = str(target["key"])
        stat = str(target["stat"])
        for ag_name, t_indices in masks.items():
            if not t_indices:
                continue
            _pool = _make_pool_fn(panels, T, t_indices)
            full_key = f"{key_base}__age{ag_name}"
            if stat == "mean":
                result[full_key] = _mean(_pool(str(target["var"])))
            elif stat == "sd":
                result[full_key] = _sd(_pool(str(target["var"])))
            elif stat == "corr":
                v0, v1 = target["vars"]
                result[full_key] = _corr(
                    _pool(str(v0)), _pool(str(v1))
                )
            elif stat == "autocorr":
                result[full_key] = _autocorr(
                    panels[str(target["var"])], t_indices
                )
            elif stat == "fraction":
                result[full_key] = _fraction(
                    _pool(str(target["var"])), target["value"]
                )
            else:
                raise ValueError(f"unknown target stat: {stat}")

    out: dict[str, float] = {}
    for k, v in result.items():
        if isinstance(v, (float, np.floating)):
            out[k] = float(v) if np.isfinite(v) else float("nan")
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_moment_fn(moment_spec: dict) -> Callable[[dict], dict[str, float]]:
    """Compile moment spec into panels -> dict[str, float]."""

    def moment_fn(panels: dict) -> dict[str, float]:
        return compute_moments_from_panels(panels, moment_spec)

    return moment_fn


def _dummy_panels_for_keys(spec: dict) -> dict[str, np.ndarray]:
    T, N = 10, 20
    names = _collect_variable_names(spec)
    if not names:
        names = ["_placeholder"]
    dummy: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for v in names:
        if v in ("discrete", "z_idx"):
            a = np.zeros((T, N), dtype=float)
            a[:, 1::2] = 1.0
            dummy[v] = a
        else:
            dummy[v] = rng.standard_normal((T, N))
    return dummy


def moment_names(moment_spec: dict) -> list[str]:
    """Deterministic list of moment keys (same order as compute_moments_from_panels)."""
    dummy = _dummy_panels_for_keys(moment_spec)
    m = compute_moments_from_panels(dummy, moment_spec)
    return list(m.keys())


def load_data_moments(path: Any, moment_spec: dict) -> dict[str, float]:
    """Load empirical moments from dict path, precomputed CSV, or microdata file."""
    if isinstance(path, dict):
        out: dict[str, float] = {}
        for k, v in path.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, (int, float, np.floating)):
                fv = float(v)
                if np.isnan(fv):
                    continue
                out[str(k)] = fv
        return out

    path_str = str(path)
    data_source = moment_spec.get("data_source", "precomputed")

    if data_source == "microdata":
        if path_str.endswith(".parquet"):
            df = pd.read_parquet(path_str)
        else:
            df = pd.read_csv(path_str)
        return compute_moments_from_dataframe(df, moment_spec)

    df = pd.read_csv(path_str)
    age_col = df.columns[0]
    moment_cols = [c for c in df.columns if c != age_col]
    col_map = moment_spec.get("column_map") or {}

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        age_idx = row[age_col]
        if pd.isna(age_idx):
            continue
        try:
            ag_name = str(int(float(age_idx)))
        except (TypeError, ValueError):
            ag_name = str(age_idx).strip()

        for col in moment_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if col in col_map:
                template = str(col_map[col])
                canonical = template.replace("{age}", ag_name)
                result[canonical] = float(val)
            else:
                result[f"{col}__age{ag_name}"] = float(val)

    return result


def compute_moments_from_dataframe(df: pd.DataFrame, spec: dict) -> dict[str, float]:
    """Wide/long microdata -> panels dict -> same keys as compute_moments_from_panels."""
    setup = spec.get("setup") or {}
    time_var = setup.get("time_var", "t")
    id_var = setup.get("id_var", "_idx")

    times = sorted(df[time_var].unique())
    agents = sorted(df[id_var].unique())
    T = len(times)
    N = len(agents)
    time_to_t = {v: i for i, v in enumerate(times)}
    agent_to_n = {v: i for i, v in enumerate(agents)}

    skip = {time_var, id_var}
    var_cols = [c for c in df.columns if c not in skip]
    panels: dict[str, np.ndarray] = {
        v: np.full((T, N), np.nan, dtype=float) for v in var_cols
    }

    for _, row in df.iterrows():
        t = time_to_t[row[time_var]]
        n = agent_to_n[row[id_var]]
        for v in var_cols:
            panels[v][t, n] = row[v]

    return compute_moments_from_panels(panels, spec)
