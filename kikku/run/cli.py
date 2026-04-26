"""CLI: ``parse_cli`` — argv → ``RunSpec`` (v3, kikku-runspec-v3)."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import warnings
from dataclasses import replace
from itertools import product
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
# Slot bundle merge (dict[str, Any] = slot name → value)
# ---------------------------------------------------------------------------


def _deep_merge(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out: dict = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(b)


def _merge_slot_bundles(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(base, overlay)


# ---------------------------------------------------------------------------
# @file resolution (§5.6)
# ---------------------------------------------------------------------------


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
        p = Path(t[1:].strip()).resolve()
        return p.read_text()
    return t


def _resolve_at_refs(value: Any, depth: int = 0, max_depth: int = 5) -> Any:
    if depth > max_depth:
        raise ValueError(
            "@file reference depth exceeded (max=5). Possible cycle."
        )
    if isinstance(value, str) and value.startswith("@"):
        path = Path(value[1:].strip()).resolve()
        if not path.is_file():
            raise ValueError(f"@file reference not found: {path}")
        text = path.read_text()
        return _resolve_at_refs(_parse_json_or_yaml(text), depth + 1, max_depth)
    if isinstance(value, dict):
        return {k: _resolve_at_refs(v, depth, max_depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_at_refs(v, depth, max_depth) for v in value]
    return value


# ---------------------------------------------------------------------------
# parse_cli
# ---------------------------------------------------------------------------

_SLOT_OVERRIDE_RE = re.compile(
    r"^\$(?P<slot>[^.]+)\.(?P<sub>[^=]+)=(?P<rest>.*)\s*$"
)


def _parse_slot_override_item(item: str) -> tuple[str, str, Any]:
    s = str(item).strip()
    m = _SLOT_OVERRIDE_RE.match(s)
    if not m:
        raise ValueError(
            f"Invalid --slot-override (expected $slot.subkey=value): {item!r}"
        )
    slot, sub, rest = m.group("slot"), m.group("sub"), m.group("rest")
    if not sub.strip() or not rest.strip():
        raise ValueError(f"Empty subkey or value in: {item!r}")
    return slot.strip(), sub.strip(), yaml.safe_load(rest.strip())


def _apply_slot_overrides_to_base(base: dict[str, Any], items: list[str]) -> None:
    for it in items:
        if not str(it).strip():
            continue
        slot, sub, val = _parse_slot_override_item(it)
        d = base.setdefault(slot, {})
        if not isinstance(d, dict):
            d = base[slot] = {}
        d[sub] = val


def _parse_resolved_dict(s: str) -> dict:
    raw = _resolve_file_or_literal(s)
    d = _parse_json_or_yaml(raw)
    if not isinstance(d, dict):
        raise ValueError(f"Expected a JSON/YAML object, got {type(d)}")
    return d


def _apply_slot_merge_log(merge_log: list) -> tuple[dict[str, Any], list[str]]:
    base: dict[str, Any] = {}
    range_axes: list[str] = []
    for g, kind, data in merge_log:
        if g != "slot":
            continue
        if kind == "override":
            parts: list = data if isinstance(data, list) else []
            _apply_slot_overrides_to_base(base, [str(p) for p in parts if str(p).strip()])
        elif kind == "spec":
            d = _parse_resolved_dict(str(data).strip() if data else "")
            d = _resolve_at_refs(d)
            base = _merge_slot_bundles(base, d)
        elif kind == "range":
            s = (str(data) if data is not None else "").strip()
            if s:
                range_axes.append(s)
    return base, range_axes


def _parse_list_payload(s: str) -> list[dict]:
    data = _parse_json_or_yaml(_resolve_file_or_literal(s))
    if not isinstance(data, list):
        raise ValueError("Range must be a JSON/YAML list of objects")
    out: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            raise ValueError("Each range element must be a dict")
        it2 = _resolve_at_refs(dict(it))
        if not isinstance(it2, dict):
            raise ValueError("After @-resolution, each range row must be a dict")
        out.append(dict(it2))
    return out


def _parse_compare(compare_str: str) -> tuple[list[dict[str, Any]], list[Any]]:
    if "=" not in compare_str:
        raise ValueError("--compare requires $slot.subkey=v1,v2,…")
    key, rhs = compare_str.split("=", 1)
    key = key.strip()
    value_strs = [x.strip() for x in rhs.split(",") if x.strip()]
    if not value_strs:
        raise ValueError("--compare: empty value list after '='")
    m = re.match(
        r"^\$(?P<slot>[^.]+)\.(?P<sub>[^=]+)\s*$",
        key,
    )
    if not m:
        raise ValueError(
            f"--compare key must be $slot.subkey, got: {key!r}"
        )
    slot, sub = m.group("slot"), m.group("sub")
    if not sub.strip():
        raise ValueError("Empty subkey in --compare")
    values = [yaml.safe_load(x) for x in value_strs]
    rows: list[dict[str, Any]] = []
    for v in values:
        rows.append({slot: {sub: v}})
    return rows, values


def parse_cli(
    name: str,
    base_spec: str,
    modes: list[str] | None = None,
    output: str = "results",
    extra_args: dict | None = None,
) -> RunSpec:
    """argv → RunSpec. v3: slot-keyed; no calibration/settings YAML reads."""
    modes_s = set(modes or [])
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        p = _build_argparser(modes_s, extra_args, name, output)
        p.print_help()
        raise SystemExit(0)

    base = Path(base_spec).resolve()

    p = _build_argparser(modes_s, extra_args, name, output)
    args = p.parse_args()
    merge_log: list = list(args.merge_log or [])

    shared_base, range_axes = _apply_slot_merge_log(merge_log)

    has_compare = getattr(args, "compare", None) is not None
    has_sweep = bool(getattr(args, "sweep", False))
    if has_compare and has_sweep:
        raise ValueError("Cannot use --compare and --sweep together.")
    if has_compare and range_axes:
        raise ValueError(
            "--compare cannot be combined with --slot-range. "
            "Use --sweep with --slot-range instead."
        )

    if has_compare:
        comp_rows, compare_values = _parse_compare(str(args.compare).strip())
        test_set = make_test(
            [_merge_slot_bundles(dict(shared_base), r) for r in comp_rows]
        )
        if len(compare_values) != len(test_set):
            raise ValueError("Internal error: compare values vs test_set length.")
        test_set = tuple(
            replace(t, label=repr(v)) for t, v in zip(test_set, compare_values)
        )
    elif range_axes:
        axes_parsed = [_parse_list_payload(ax) for ax in range_axes]
        bundles: list[dict[str, Any]] = []
        for row_tuple in product(*axes_parsed):
            m = dict(shared_base)
            for part in row_tuple:
                m = _merge_slot_bundles(m, part)
            bundles.append(m)
        test_set = make_test(bundles)
    else:
        test_set = make_test([shared_base] if shared_base else [{}])

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
# merge log (argv order, slot-only v3)
# ---------------------------------------------------------------------------


def _new_slot_merge_event_action(kind: str):
    class SlotMergeEventAction(argparse.Action):
        def __init__(self, *a, k=kind, **kw):
            self._k = k
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
                namespace.merge_log.append(("slot", "override", parts))
            else:
                s = values[0] if values is not None else ""
                if not s and option_string and "=" in option_string:
                    s = option_string.split("=", 1)[1]
                s = (s or "").strip()
                namespace.merge_log.append(("slot", self._k, s))

    return SlotMergeEventAction


def _parse_kv(item: str) -> tuple:
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got: {item!r}")
    key, val = item.split("=", 1)
    return key.strip(), yaml.safe_load(val.strip())


def _build_argparser(
    modes_s: set[str], extra_args: dict | None, name: str, default_output: str
) -> argparse.ArgumentParser:
    A = _new_slot_merge_event_action
    p = argparse.ArgumentParser(description=f"Run {name} model (v3 CLI)")
    p.set_defaults(merge_log=[])
    p.add_argument(
        "--slot-override",
        action=A("override"),
        help="Slot override: $name.subkey=value (repeatable, merge order with --slot-spec).",
    )
    p.add_argument(
        "--slot-spec",
        action=A("spec"),
        help="Slot bundle as JSON/YAML or @file.",
    )
    p.add_argument(
        "--slot-range",
        action=A("range"),
        help="One sweep axis: list of bundle dicts, JSON/YAML or @file. Repeat for Cartesian product.",
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
            metavar="$SLOT.SUBKEY=VAL,VAL,…",
            help="Single-axis compare (sugar for one slot-range on scalars).",
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
