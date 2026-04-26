"""CLI: ``parse_cli`` — argv → ``RunSpec`` (v2, kikku-runspec-v2)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .make_test import make_test
from .types import RunSpec, SimSpec

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def parse_key_value(raw_list: list | None) -> dict:
    """Parse ``['key=value', ...]`` into ``{key: typed_value}``."""
    result: dict = {}
    for item in (raw_list or []):
        k, v = _parse_kv(item)
        result[k] = v
    return result


def make_run_dir(base_dir, tag=None):
    """``base_dir/YYYY-MM-DD/NNN/``; MPI-broadcast the path to all ranks."""
    from datetime import date
    from .mpi import bcast_item, get_comm, rank_size

    if tag is not None:
        warnings.warn(
            "make_run_dir 'tag' parameter is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    comm = get_comm()
    rank, _ = rank_size(comm)
    run_dir_str = None
    if rank == 0:
        base = Path(base_dir)
        today = date.today().isoformat()
        day_dir = base / today
        n = 1
        while True:
            run_dir = day_dir / f"{n:03d}"
            if not run_dir.exists():
                break
            n += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_str = str(run_dir)
    return bcast_item(run_dir_str, comm, root=0)


# ---------------------------------------------------------------------------
# parse_cli
# ---------------------------------------------------------------------------


def parse_cli(
    name: str,
    base_spec: str,
    modes: list[str] | None = None,
    output: str = "results",
    extra_args: dict | None = None,
) -> RunSpec:
    """Build a ``RunSpec`` from command-line arguments (§14)."""
    modes_s = set(modes or [])
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        p = _build_argparser(modes_s, extra_args, name, output)
        p.print_help()
        raise SystemExit(0)

    base = Path(base_spec).resolve()
    _cp, _ct, params_keys, _ = _load_params_side(base)
    _sp, _st, settings_keys, _ = _load_settings_side(base)
    if params_keys & settings_keys:
        amb = params_keys & settings_keys
        raise ValueError(
            f"Base spec {base} has non-empty params_keys ∩ settings_keys: "
            f"{sorted(amb)}. Fix the calibration/settings YAMLs."
        )

    p = _build_argparser(modes_s, extra_args, name, output)
    args = p.parse_args()
    merge_log: list = list(args.merge_log or [])

    p_m, s_m, m_m, p_r, s_r, m_r = _apply_merge_log(
        merge_log, params_keys, settings_keys
    )

    for k in p_m:
        if "." in k:
            raise ValueError(
                f"Dotted key {k!r} is a methods path; use --methods-override or "
                f"--methods-spec."
            )
        _validate_tier_key(k, "params", params_keys, settings_keys)
    for k in s_m:
        if "." in k:
            raise ValueError(
                f"Dotted key {k!r} is a methods path; use --methods-override or "
                f"--methods-spec."
            )
        _validate_tier_key(k, "settings", params_keys, settings_keys)
    for t in m_m:
        if len(t) < 3:
            raise ValueError(
                f"Method path {'.'.join(t)} must have at least three "
                f"dot-separated parts."
            )

    p_r, s_r, m_r = _validate_range_contents(
        p_r, s_r, m_r, params_keys, settings_keys
    )

    has_compare = getattr(args, "compare", None) is not None
    has_sweep = bool(getattr(args, "sweep", False))
    if has_compare and has_sweep:
        raise ValueError("Cannot use --compare and --sweep together.")
    if has_compare and (p_r is not None or s_r is not None or m_r is not None):
        raise ValueError(
            "--compare cannot be combined with --*-range. "
            "Use --sweep with --*-range instead."
        )

    p_dim, m_dim, s_dim, compare_values = _build_dimensions(
        has_compare=has_compare,
        compare_str=getattr(args, "compare", None),
        p_m=p_m,
        s_m=s_m,
        m_m=m_m,
        p_r_raw=p_r,
        s_r_raw=s_r,
        m_r_raw=m_r,
        params_keys=params_keys,
        settings_keys=settings_keys,
    )

    test_set = make_test(params=p_dim, methods=m_dim, settings=s_dim)
    if has_compare and compare_values is not None:
        if len(compare_values) != len(test_set):
            raise ValueError("Internal error: compare values vs test_set length.")
        test_set = tuple(
            replace(t, label=repr(v)) for t, v in zip(test_set, compare_values)
        )

    # Mode (§14): --compare → "compare";
    # --sweep OR any multi-row test_set (incl. implicit via --*-range) → "sweep";
    # else "single". Compare + --*-range is rejected before we get here.
    if has_compare:
        mode = "compare"
    elif has_sweep or len(test_set) > 1:
        mode = "sweep"
    else:
        mode = "single"

    if "sweep" not in modes_s and has_sweep:
        raise ValueError("Sweep is not available for this runner (modes=…).")
    if "compare" not in modes_s and has_compare:
        raise ValueError("Compare is not available for this runner (modes=…).")
    if "simulate" not in modes_s and getattr(args, "simulate", False):
        raise ValueError("Simulate is not available for this runner (modes=…).")

    if getattr(args, "simulate", False):
        sim = SimSpec(
            n_sim=args.n_sim,
            seed=args.seed,
            plots=getattr(args, "plots", False),
        )
    else:
        sim = None

    extra: dict = {}
    if extra_args:
        for flag in extra_args:
            attr = flag.lstrip("-").replace("-", "_")
            extra[attr] = getattr(args, attr, None)

    out_dir = Path(make_run_dir(args.output_dir))
    return RunSpec(
        name=name,
        base_spec=base,
        output_dir=out_dir,
        run_tag=args.run_tag,
        test_set=test_set,
        params_keys=frozenset(params_keys),
        settings_keys=frozenset(settings_keys),
        sweep_runs=getattr(args, "sweep_runs", 1)
        if "sweep" in modes_s
        else 1,
        warmup=getattr(args, "warmup", True),
        sim=sim,
        mode=mode,
        extra_args=extra,
        verbose=args.verbose,
        trace=args.trace,
        mpi=getattr(args, "mpi", False),
        gpu=getattr(args, "gpu", False),
        skip_egm_plots=getattr(args, "skip_egm_plots", False),
        csv_export=getattr(args, "csv_export", False),
    )


# ---------------------------------------------------------------------------
# merge log (argv order, §13.4)
# ---------------------------------------------------------------------------


def _new_merge_event_action(group: str, kind: str):
    class MergeEventAction(argparse.Action):
        def __init__(self, *a, g=group, k=kind, **kw):
            self._g, self._k = g, k
            if kind == "override" and "nargs" not in kw:
                kw["nargs"] = "*"
            if kind in ("spec", "range") and "nargs" not in kw:
                kw["nargs"] = 1
            super().__init__(*a, **kw)

        def __call__(self, parser, namespace, values, option_string=None):
            if not hasattr(namespace, "merge_log") or namespace.merge_log is None:
                namespace.merge_log = []
            if self._k == "override":
                parts: list = []
                if values:
                    parts = [str(v) for v in values if str(v).strip()]
                if not parts and option_string and "=" in option_string:
                    rest = option_string.split("=", 1)[1]
                    if rest:
                        parts = [rest]
                namespace.merge_log.append((self._g, "override", parts))
            else:
                s = values[0] if values is not None else ""
                if not s and option_string and "=" in option_string:
                    s = option_string.split("=", 1)[1]
                s = (s or "").strip()
                namespace.merge_log.append((self._g, self._k, s))

    return MergeEventAction


def _apply_merge_log(
    merge_log: list,
    params_keys: set[str],
    settings_keys: set[str],
) -> tuple[dict, dict, dict, str | None, str | None, str | None]:
    p_m: dict = {}
    s_m: dict = {}
    m_m: dict = {}
    p_r, s_r, m_r = None, None, None
    for g, kind, data in merge_log:
        if g == "params" and kind == "override":
            for item in data:
                if not str(item).strip():
                    continue
                k, v = _parse_kv(str(item))
                _validate_tier_key(k, "params", params_keys, settings_keys)
                p_m[k] = v
        elif g == "settings" and kind == "override":
            for item in data:
                if not str(item).strip():
                    continue
                k, v = _parse_kv(str(item))
                _validate_tier_key(k, "settings", params_keys, settings_keys)
                s_m[k] = v
        elif g == "methods" and kind == "override":
            for item in data:
                if not str(item).strip():
                    continue
                path, tag = _split_method_path_tag(str(item))
                t = tuple(path.split("."))
                if len(t) < 3:
                    raise ValueError(
                        f"Method path {path!r} must have at least three "
                        f"dot-separated parts."
                    )
                m_m[t] = tag
        elif g == "params" and kind == "spec":
            d = _parse_resolved_dict(str(data))
            for kk, vv in d.items():
                if "." in str(kk):
                    raise ValueError(
                        f"Dotted key {kk!r} in --params-spec; use --methods-spec for "
                        f"method slots."
                    )
                _validate_tier_key(str(kk), "params", params_keys, settings_keys)
            p_m.update(d)
        elif g == "settings" and kind == "spec":
            d = _parse_resolved_dict(str(data))
            for kk, vv in d.items():
                if "." in str(kk):
                    raise ValueError(
                        f"Dotted key {kk!r} in --settings-spec; use --methods-spec for "
                        f"method slots."
                    )
                _validate_tier_key(str(kk), "settings", params_keys, settings_keys)
            s_m.update(d)
        elif g == "methods" and kind == "spec":
            d = _parse_resolved_dict(str(data))
            m_m.update(
                {
                    _key_to_method_tuple(a): (str(b) if b is not None else b)
                    for a, b in d.items()
                }
            )
        elif g == "params" and kind == "range":
            p_r = _resolve_file_or_literal(data)
        elif g == "settings" and kind == "range":
            s_r = _resolve_file_or_literal(data)
        elif g == "methods" and kind == "range":
            m_r = _resolve_file_or_literal(data)
    return p_m, s_m, m_m, p_r, s_r, m_r


def _key_to_method_tuple(kk) -> tuple[str, ...]:
    t = kk if isinstance(kk, tuple) else tuple(str(kk).split("."))
    if len(t) < 3:
        raise ValueError(
            f"Method key {'.'.join(t)} must have at least three dot-separated parts."
        )
    return t


def _parse_resolved_dict(s: str) -> dict:
    raw = _resolve_file_or_literal(s)
    d = _parse_json_or_yaml(raw)
    if not isinstance(d, dict):
        raise ValueError(f"Expected a JSON/YAML object, got {type(d)}")
    return d


def _validate_range_contents(
    p_r, s_r, m_r, params_keys, settings_keys
) -> tuple:
    if p_r is not None:
        for item in _parse_list_payload(p_r):
            for kk in item:
                _validate_tier_key(str(kk), "params", params_keys, settings_keys)
    if s_r is not None:
        for item in _parse_list_payload(s_r):
            for kk in item:
                _validate_tier_key(str(kk), "settings", params_keys, settings_keys)
    if m_r is not None:
        for item in _parse_list_payload(m_r):
            for kk in item:
                _ = _key_to_method_tuple(kk)
    return p_r, s_r, m_r


def _build_dimensions(
    *,
    has_compare: bool,
    compare_str: str | None,
    p_m: dict,
    s_m: dict,
    m_m: dict,
    p_r_raw: str | None,
    s_r_raw: str | None,
    m_r_raw: str | None,
    params_keys: set,
    settings_keys: set,
) -> tuple[
    list[dict] | None,
    list[dict | None] | None,
    list[dict] | None,
    list[Any] | None,
]:
    """Return (p_dim, m_dim, s_dim, compare_value_list or None)."""
    compare_vals: list[Any] | None = None

    if has_compare:
        if not compare_str or "=" not in compare_str:
            raise ValueError("--compare requires key=v1,v2,…")
        key, rhs = compare_str.split("=", 1)
        key = key.strip()
        value_strs = [x.strip() for x in rhs.split(",") if x.strip()]
        values = [yaml.safe_load(x) for x in value_strs]
        if "." in key:
            tier = "methods"
        elif key in params_keys and key in settings_keys:
            raise ValueError(
                f"Key {key!r} is in both params and settings (this should be impossible)."
            )
        elif key in params_keys:
            tier = "params"
        elif key in settings_keys:
            tier = "settings"
        else:
            raise ValueError(
                f"Unknown key {key!r}; not a params/settings name and not a dotted method path."
            )
        if tier == "params":
            p_dim = [{**p_m, key: v} for v in values]
            m_dim = [m_m] if m_m else [None]
            s_dim = [s_m]
        elif tier == "settings":
            s_dim = [{**s_m, key: v} for v in values]
            p_dim = [p_m]
            m_dim = [m_m] if m_m else [None]
        else:
            tkey = _key_to_method_tuple(key)
            m_dim = [
                {**m_m, tkey: (str(v) if v is not None else v)} for v in values
            ]
            p_dim = [p_m]
            s_dim = [s_m]
        compare_vals = list(values)
        return p_dim, m_dim, s_dim, compare_vals

    p_dim: list[dict] = (
        [dict(p_m)]
        if p_r_raw is None
        else [{**p_m, **row} for row in _parse_list_payload(p_r_raw)]
    )
    s_dim: list[dict] = (
        [dict(s_m)]
        if s_r_raw is None
        else [{**s_m, **row} for row in _parse_list_payload(s_r_raw)]
    )
    if m_r_raw is None:
        m_dim: list[dict | None] = [m_m] if m_m else [None]
    else:
        base: dict = dict(m_m) if m_m else {}
        m_dim = []
        for d in _parse_list_payload(m_r_raw):
            nd: dict = {}
            for a, b in d.items():
                nd[_key_to_method_tuple(a)] = str(b) if b is not None else b
            merged: dict = {**base, **nd}
            m_dim.append(None if not merged else merged)
    return p_dim, m_dim, s_dim, None


# ---------------------------------------------------------------------------
# YAML, literals, base_spec resolution (§14 step 2)
# ---------------------------------------------------------------------------


def _load_params_side(base: Path) -> tuple[Path, list[Path], set[str], dict]:
    cands = [base / "calibration.yaml", base / "calibration" / "main.yaml"]
    for p in cands:
        if p.is_file():
            raw = _load_yaml_file(p)
            d = _param_dict_from_calib_file(raw)
            return p, cands, set(d.keys()), d
    raise ValueError(
        f"No calibration file found. Tried: "
        f"{', '.join(str(t) for t in cands)}"
    )


def _load_settings_side(base: Path) -> tuple[Path, list[Path], set[str], dict]:
    cands = [base / "settings.yaml", base / "settings" / "default.yaml"]
    for p in cands:
        if p.is_file():
            raw = _load_yaml_file(p)
            d = _settings_dict_from_file(raw)
            return p, cands, set(d.keys()), d
    raise ValueError(
        f"No settings file found. Tried: {', '.join(str(t) for t in cands)}"
    )


def _load_yaml_file(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _param_dict_from_calib_file(raw: dict) -> dict:
    if "calibration" in raw and isinstance(raw["calibration"], dict):
        return dict(raw["calibration"])
    return dict(raw) if raw else {}


def _settings_dict_from_file(raw: dict) -> dict:
    if "settings" in raw and isinstance(raw["settings"], dict):
        return dict(raw["settings"])
    return dict(raw) if raw else {}


def _parse_kv(item: str) -> tuple:
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got: {item!r}")
    key, val = item.split("=", 1)
    return key.strip(), yaml.safe_load(val.strip())


def _split_method_path_tag(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(
            f"Method override must be path=tag (≥3 dotted path), got: {item!r}"
        )
    path, tag = item.rsplit("=", 1)
    return path.strip(), str(tag).strip()


def _parse_json_or_yaml(content: str) -> Any:
    s = content.strip()
    if not s:
        raise ValueError("Empty spec literal")
    if s[0] in "[{":
        return json.loads(s)
    return yaml.safe_load(s)


def _resolve_file_or_literal(s: str) -> str:
    t = s.strip()
    if t.startswith("@"):
        p = Path(t[1:]).resolve()
        return p.read_text()
    return t


def _parse_list_payload(s: str) -> list[dict]:
    data = _parse_json_or_yaml(_resolve_file_or_literal(s))
    if not isinstance(data, list):
        raise ValueError("Range must be a JSON/YAML list of objects")
    out: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            raise ValueError("Each range element must be a dict")
        out.append(dict(it))
    return out


def _validate_tier_key(
    k: str, tier: str, params_keys: set[str], settings_keys: set[str]
) -> None:
    is_dotted = "." in k
    if is_dotted:
        n = len(k.split("."))
        if n < 3:
            raise ValueError(
                f"Key {k!r} is not a valid method path (need at least three "
                f"dot-separated parts)."
            )
        if tier in ("params", "settings"):
            raise ValueError(
                f"Dotted key {k!r} belongs in --methods-override or --methods-spec, "
                f"not --{tier}."
            )
        return
    in_p, in_s = k in params_keys, k in settings_keys
    if tier == "params":
        if in_s and not in_p:
            raise ValueError(
                f"{k!r} is in settings_keys; use --settings-override or --settings-spec"
            )
        if not in_p:
            raise ValueError(
                f"Unknown params key {k!r} (not in base calibration and not a "
                f"method path)."
            )
    elif tier == "settings":
        if in_p and not in_s:
            raise ValueError(
                f"{k!r} is in params_keys; use --params-override or --params-spec"
            )
        if not in_s:
            raise ValueError(
                f"Unknown settings key {k!r} (not in base settings and not a method path)."
            )


def _build_argparser(
    modes_s: set[str], extra_args: dict | None, name: str, default_output: str
) -> argparse.ArgumentParser:
    M = _new_merge_event_action
    p = argparse.ArgumentParser(description=f"Run {name} model (v2 CLI)")
    p.set_defaults(merge_log=[])
    p.add_argument(
        "--params-override",
        action=M("params", "override"),
        help="params override: key=value (repeatable, argv merge order with --params-spec).",
    )
    p.add_argument(
        "--params-spec",
        action=M("params", "spec"),
        help="Params bundle as JSON/YAML or @file.",
    )
    p.add_argument(
        "--params-range",
        action=M("params", "range"),
        help="Params sweep dimension: list of dicts, JSON/YAML or @file.",
    )
    p.add_argument(
        "--settings-override",
        action=M("settings", "override"),
    )
    p.add_argument(
        "--settings-spec",
        action=M("settings", "spec"),
    )
    p.add_argument(
        "--settings-range",
        action=M("settings", "range"),
    )
    p.add_argument(
        "--methods-override",
        action=M("methods", "override"),
    )
    p.add_argument(
        "--methods-spec",
        action=M("methods", "spec"),
    )
    p.add_argument(
        "--methods-range",
        action=M("methods", "range"),
    )
    p.add_argument("--output-dir", type=str, default=default_output)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--trace", action="store_true", default=False)
    if "compare" in modes_s:
        p.add_argument(
            "--compare",
            type=str,
            default=None,
            metavar="KEY=VAL,VAL,…",
            help="Single-axis compare (sugar for one *-range).",
        )
    if "sweep" in modes_s:
        p.add_argument("--sweep", action="store_true", default=False)
        p.add_argument("--sweep-runs", type=int, default=3)
        p.add_argument(
            "--warmup",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    if "simulate" in modes_s:
        p.add_argument("--simulate", action="store_true", default=False)
        p.add_argument("--n-sim", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=42)
    if "mpi" in modes_s:
        p.add_argument("--mpi", action="store_true", default=False)
    if "gpu" in modes_s:
        p.add_argument("--gpu", action="store_true", default=False)
    if "plots" in modes_s:
        p.add_argument("--plots", action="store_true", default=False)
        p.add_argument("--skip-egm-plots", action="store_true", default=False)
        p.add_argument("--csv-export", action="store_true", default=False)
    if extra_args:
        for flag, kwargs in extra_args.items():
            p.add_argument(flag, **kwargs)
    return p
