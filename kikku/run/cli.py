"""CLI: ``parse_cli`` — argv → ``RunSpec`` (v4, kikku-runspec-v4 axis-CLI).

Pure delivery layer — no semantic composition. Slot-payload merging into a
recipe's defaults lives entirely in ``spec_factory.maker``. Kikku's
``parse_cli`` only does mechanical operations (v4 spec §1):

- argparse + ``@file`` resolution with cycle-depth cap
- JSON / YAML literal decoding
- dot-path → nested-dict desugaring (``$a.b.c=v`` → ``{a: {b: {c: v}}}``)
- argv-order, last-writer-wins assembly of one row's slot bundle
- Cartesian product across ``--slot-range`` axes
- ``TestSpec`` / ``RunSpec`` construction

Composition is visible in ``parse_cli``::

    args        = _build_argparser(...).parse_args()
    base, axes  = _execute_merge_log(args.merge_log)        # argv replay
    if has_compare:  axes, labels = _compare_to_axis(...)
    test_set    = _expand_test_set(base, axes, labels)      # Cartesian
    return RunSpec(...)
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml

from .make_test import make_test
from .types import RunSpec, SimSpec, TestSpec

# ---------------------------------------------------------------------------
# Pure primitives
# ---------------------------------------------------------------------------


def _deep_merge(a: Any, b: Any) -> Any:
    """Recursive dict merge; ``b`` wins on collision; non-dicts replace."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: dict = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(b)


def _set_path(d: dict, path: list[str], value: Any) -> None:
    """Walk ``d`` along ``path``; assign ``value`` at the leaf.

    Type-collision rule (§3.3): if a key along the path holds a non-dict,
    it is overwritten with an empty dict so the deeper write proceeds.
    Argv-order, last-writer-wins.
    """
    if not path:
        raise ValueError("_set_path: empty path")
    cur: dict = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def _parse_json_or_yaml(content: str) -> Any:
    s = content.strip()
    if not s:
        raise ValueError("Empty literal")
    if s[0] in "[{":
        return json.loads(s)
    return yaml.safe_load(s)


def _resolve_file_or_literal(s: str) -> str:
    """``@path`` → file contents; otherwise return the string verbatim."""
    t = s.strip()
    if t.startswith("@"):
        p = Path(t[1:].strip()).resolve()
        return p.read_text()
    return t


def _resolve_at_refs(value: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """Recursively resolve ``@file`` references in nested values (§5.6)."""
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
# Slot-path parsing  ($slot.k1[.k2…])
# ---------------------------------------------------------------------------

_SLOT_HEAD_RE = re.compile(r"^\$(?P<slot>[A-Za-z_][A-Za-z0-9_]*)")


def _parse_slot_path(s: str) -> tuple[str, list[str]]:
    """``$slot.k1.k2.k3`` → ``('slot', ['k1', 'k2', 'k3'])``.

    Raises ``ValueError`` for missing leading ``$`` or empty slot name.
    Empty path (``$slot``) returns ``('slot', [])`` — callers decide if
    zero-length paths are acceptable in their context.
    """
    s = s.strip()
    m = _SLOT_HEAD_RE.match(s)
    if not m:
        raise ValueError(
            f"Slot path must start with $<name>, got: {s!r}"
        )
    slot = m.group("slot")
    rest = s[m.end():]
    if not rest:
        return slot, []
    if not rest.startswith("."):
        raise ValueError(
            f"Slot path must use '$slot.key[.key…]' form, got: {s!r}"
        )
    parts = [p for p in rest[1:].split(".")]
    if any(not p for p in parts):
        raise ValueError(f"Empty path component in: {s!r}")
    return slot, parts


def _parse_slot_override(s: str) -> tuple[str, list[str], Any]:
    """``$slot.k1[.k2…]=value`` → ``(slot, path, value)``.

    Rejects zero-path forms (``$slot=v``); use ``--slot-spec`` for that.
    Value parsed via ``yaml.safe_load`` (YAML 1.1 type coercion applies —
    ``yes``/``on`` become ``True`` etc.; documented in --help).
    """
    if "=" not in s:
        raise ValueError(
            f"Invalid --slot-override (expected $slot.subkey=value): {s!r}"
        )
    lhs, rhs = s.split("=", 1)
    try:
        slot, path = _parse_slot_path(lhs.strip())
    except ValueError as e:
        raise ValueError(
            f"Invalid --slot-override LHS ({e}): {s!r}"
        ) from None
    if not path:
        raise ValueError(
            f"Invalid --slot-override (zero-path forms not allowed; "
            f"use --slot-spec to bind a whole slot value): {s!r}"
        )
    if not rhs.strip():
        raise ValueError(f"Invalid --slot-override (empty value): {s!r}")
    return slot, path, yaml.safe_load(rhs.strip())


# ---------------------------------------------------------------------------
# RangeAxis — unified shape for axis-form and bundle-list ranges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeAxis:
    """One sweep axis (§7.3).

    ``axis`` kind: scalar/dict values along a single slot path.
    ``bundle_list`` kind: verbatim row bundles (rows enumerated as dicts).
    """

    kind: Literal["axis", "bundle_list"]
    slot: str | None
    path: tuple[str, ...] | None
    rows: tuple[Any, ...]


def _parse_axis_range(raw: str) -> RangeAxis:
    """``$slot[.path]=[v1, v2, …]`` or ``$slot[.path]=@file.yaml`` → axis ``RangeAxis``.

    Spec §4.1 + §4.5: zero-path form (``$slot=[...]``) is allowed for
    whole-slot replacement axes (e.g. method-block payloads); deep-path form
    (``$slot.k1.k2=[...]``) for scalar/dict axes inside a slot.
    """
    if "=" not in raw:
        raise ValueError(
            "Axis-form --slot-range expects $slot[.path]=[v1, v2, ...]; "
            f"got: {raw!r}"
        )
    lhs, rhs = raw.split("=", 1)
    slot, path = _parse_slot_path(lhs.strip())
    rhs = rhs.strip()
    rows_text = _resolve_file_or_literal(rhs) if rhs.startswith("@") else rhs
    rows = _parse_json_or_yaml(rows_text)
    if not isinstance(rows, list):
        raise ValueError(
            "Axis-form --slot-range RHS must be a list; "
            f"got: {type(rows).__name__}"
        )
    if not rows:
        raise ValueError(f"Axis cannot be empty: {raw!r}")
    rows = [_resolve_at_refs(r) for r in rows]
    return RangeAxis(kind="axis", slot=slot, path=tuple(path), rows=tuple(rows))


def _parse_bundle_list_range(raw: str) -> RangeAxis:
    """JSON/YAML list of bundle dicts → ``bundle_list`` RangeAxis (§4.4)."""
    data = _parse_json_or_yaml(raw)
    if not isinstance(data, list):
        raise ValueError(
            "Bundle-list --slot-range must be a JSON/YAML list of objects"
        )
    rows: list[dict] = []
    for it in data:
        if not isinstance(it, dict):
            raise ValueError("Each bundle-list element must be a dict")
        rows.append(_resolve_at_refs(dict(it)))
    if not rows:
        raise ValueError("Bundle-list cannot be empty")
    return RangeAxis(kind="bundle_list", slot=None, path=None, rows=tuple(rows))


def _parse_slot_range(s: str) -> RangeAxis:
    """Dispatch on leading ``$`` (§7.3)."""
    raw = _resolve_file_or_literal(s).strip()
    if raw.startswith("$"):
        return _parse_axis_range(raw)
    return _parse_bundle_list_range(raw)


def _compare_to_axis(s: str) -> tuple[RangeAxis, tuple[str, ...]]:
    """``--compare $slot.path=v1,v2,…`` → (axis, str-labels) (§5.2)."""
    if "=" not in s:
        raise ValueError("--compare requires $slot.subkey=v1,v2,…")
    lhs, rhs = s.split("=", 1)
    slot, path = _parse_slot_path(lhs.strip())
    if not path:
        raise ValueError(
            f"--compare key must be $slot.subkey[.…], got: {lhs!r}"
        )
    value_strs = [x.strip() for x in rhs.split(",") if x.strip()]
    if not value_strs:
        raise ValueError("--compare: empty value list after '='")
    values = [yaml.safe_load(x) for x in value_strs]
    axis = RangeAxis(
        kind="axis", slot=slot, path=tuple(path), rows=tuple(values)
    )
    labels = tuple(str(v) for v in values)
    return axis, labels


# ---------------------------------------------------------------------------
# Argv-order replay  →  shared_base + range axes
# ---------------------------------------------------------------------------


def _apply_override_event(base: dict, parts: list[str]) -> None:
    """Replay one ``--slot-override`` event onto ``base``.

    ``merge_log`` carries the raw argv strings (not pre-parsed
    ``(slot, path, value)`` triples) so the log stays inspectable as the
    user typed it. Parsing happens here, at replay time.
    """
    for p in parts:
        if not str(p).strip():
            continue
        slot, path, value = _parse_slot_override(str(p))
        bundle = base.setdefault(slot, {})
        if not isinstance(bundle, dict):
            bundle = {}
            base[slot] = bundle
        _set_path(bundle, list(path), value)


def _apply_spec_event(base: dict, raw: str) -> dict:
    """Returns the new base (deep-merged); does not mutate the input."""
    payload = _parse_json_or_yaml(_resolve_file_or_literal(raw))
    if not isinstance(payload, dict):
        raise ValueError(
            f"--slot-spec must resolve to a JSON/YAML object, got {type(payload).__name__}"
        )
    payload = _resolve_at_refs(payload)
    return _deep_merge(base, payload)


def _execute_merge_log(
    merge_log: list,
) -> tuple[dict[str, Any], list[RangeAxis]]:
    """Argv-order replay of all `--slot-*` events.

    Returns ``(shared_base, range_axes)``. ``shared_base`` is the result of
    overlaying every ``--slot-override`` and ``--slot-spec`` event in argv
    order with last-writer-wins, key-by-key deep-merge for dict-typed values.
    Each ``--slot-range`` event becomes one ``RangeAxis`` in argv order.
    """
    base: dict[str, Any] = {}
    axes: list[RangeAxis] = []
    for group, kind, data in merge_log:
        if group != "slot":
            continue
        if kind == "override":
            parts = data if isinstance(data, list) else []
            _apply_override_event(base, [str(p) for p in parts])
        elif kind == "spec":
            base = _apply_spec_event(base, str(data) if data else "")
        elif kind == "range":
            s = (str(data) if data is not None else "").strip()
            if s:
                axes.append(_parse_slot_range(s))
    return base, axes


# ---------------------------------------------------------------------------
# Cartesian expansion + label derivation
# ---------------------------------------------------------------------------


def _axis_diff_label(axes: list[RangeAxis], combo: tuple) -> str:
    """`path1=v1,path2=v2,…` from axis-kind elements only (§7.4).

    Zero-path axes (whole-slot replacement) contribute ``$slot=str(value)``.
    """
    parts: list[str] = []
    for axis, value in zip(axes, combo):
        if axis.kind != "axis":
            continue
        full_path = ".".join([axis.slot or "", *(axis.path or ())])
        parts.append(f"{full_path}={value}")
    return ",".join(parts)


def _row_bundle(
    base: dict, axes: list[RangeAxis], combo: tuple
) -> dict[str, Any]:
    bundle = copy.deepcopy(base)
    for axis, value in zip(axes, combo):
        if axis.kind == "axis":
            assert axis.slot is not None
            if not axis.path:
                bundle[axis.slot] = copy.deepcopy(value)
            else:
                if not isinstance(bundle.get(axis.slot), dict):
                    bundle[axis.slot] = {}
                _set_path(bundle[axis.slot], list(axis.path), value)
        else:
            bundle = _deep_merge(bundle, value)
    return bundle


def _expand_test_set(
    base: dict[str, Any],
    axes: list[RangeAxis],
    compare_labels: tuple[str, ...] | None = None,
) -> tuple[TestSpec, ...]:
    """Cartesian product of axes; assemble each row's slot bundle + label."""
    if not axes:
        return tuple(make_test([base]))
    rows: list[TestSpec] = []
    combos = list(product(*[axis.rows for axis in axes]))
    if compare_labels is not None:
        if len(combos) != len(compare_labels):
            raise ValueError(
                "Internal: compare label count != axis row count "
                f"({len(compare_labels)} vs {len(combos)})"
            )
    for i, combo in enumerate(combos):
        bundle = _row_bundle(base, axes, combo)
        label = (
            compare_labels[i]
            if compare_labels is not None
            else _axis_diff_label(axes, combo)
        )
        rows.append(TestSpec(slots=bundle, label=label))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Public helpers + run-dir
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
# parse_cli  — orchestrator
# ---------------------------------------------------------------------------


def parse_cli(
    name: str,
    base_spec: str,
    modes: list[str] | None = None,
    output: str = "results",
    extra_args: dict | None = None,
) -> RunSpec:
    """argv → RunSpec (v4 axis-CLI)."""
    modes_s = set(modes or [])
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        p = _build_argparser(modes_s, extra_args, name, output)
        p.print_help()
        raise SystemExit(0)

    base = Path(base_spec).resolve()

    p = _build_argparser(modes_s, extra_args, name, output)
    args = p.parse_args()
    merge_log: list = list(args.merge_log or [])

    shared_base, range_axes = _execute_merge_log(merge_log)

    has_compare = getattr(args, "compare", None) is not None
    has_sweep = bool(getattr(args, "sweep", False))

    if has_compare and has_sweep:
        raise ValueError("Cannot use --compare and --sweep together.")
    if has_compare and range_axes:
        raise ValueError(
            "--compare cannot be combined with --slot-range. "
            "Use --sweep with --slot-range instead."
        )
    if has_sweep and not range_axes:
        raise ValueError(
            "--sweep requires at least one --slot-range."
        )

    compare_labels: tuple[str, ...] | None = None
    if has_compare:
        compare_axis, compare_labels = _compare_to_axis(str(args.compare).strip())
        range_axes = [compare_axis]

    test_set = _expand_test_set(shared_base, range_axes, compare_labels)

    if has_compare:
        mode = "compare"
    elif range_axes:
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
# argparse plumbing — merge_log builder + argparser
# ---------------------------------------------------------------------------


def _new_slot_merge_event_action(kind: str):
    """argparse Action that appends ('slot', kind, data) to namespace.merge_log
    in argv order. Kept as a closure so add_argument calls stay terse."""

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
    p = argparse.ArgumentParser(description=f"Run {name} model (v4 CLI)")
    p.set_defaults(merge_log=[])
    p.add_argument(
        "--slot-override",
        action=A("override"),
        help=(
            "Slot override using a dotted path: $name.k1[.k2.…]=value "
            "(repeatable; merged in argv order with --slot-spec). "
            "Value parsed by yaml.safe_load (YAML 1.1: yes/on/y → True, "
            "no/off/n → False; quote string-literal booleans if intended)."
        ),
    )
    p.add_argument(
        "--slot-spec",
        action=A("spec"),
        help=(
            "Slot bundle as JSON/YAML object or @file. Top-level keys are "
            "slot names; values mirror the slot's YAML structure. Merged in "
            "argv order with --slot-override (last-writer-wins, deep). "
            "Scalars parsed via yaml.safe_load (YAML 1.1: yes/on/y → True, "
            "no/off/n → False; quote string-literal booleans if intended)."
        ),
    )
    p.add_argument(
        "--slot-range",
        action=A("range"),
        help=(
            "One sweep axis. Two forms: "
            "(1) AXIS form — $slot.path=[v1, v2, …] declares one axis along "
            "a slot path, list elements are values at that path; "
            "(2) BUNDLE-LIST form — JSON/YAML list of bundle dicts (or @file). "
            "Repeat the flag for Cartesian product across axes. "
            "Scalars parsed via yaml.safe_load (YAML 1.1 booleans apply, see "
            "--slot-override). Float labels normalize trailing zeros: "
            "[0.05, 0.10] → labels '0.05' and '0.1'."
        ),
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
            metavar="$SLOT.PATH=V1,V2,…",
            help=(
                "Sugar for a single-axis --slot-range with str() row labels "
                "(unquoted in tables). Same dotted-path grammar as --slot-override."
            ),
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
