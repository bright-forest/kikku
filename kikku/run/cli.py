"""CLI entry point for DDSL model runners.

One deep module: ``parse_run`` absorbs argparse, YAML loading, override
tier enforcement, and output-directory creation.  Returns a frozen
``RunSpec`` — the single object each example's ``main()`` needs.

No model-specific knowledge.  No mpi4py import.
"""

from __future__ import annotations

import argparse
import warnings
import yaml
from dataclasses import dataclass
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# RunSpec — flat, frozen, 25+ fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Frozen specification for a single model run.

    Built by ``parse_run`` from CLI arguments + YAML files.
    Examples receive this as their sole configuration object.

    Fields are grouped by function:

    Identity
        name, model_dir

    Parameter tiers (precedence: CLI > override-file > base YAML)
        calib       — calibration parameters (economic: beta, r, tau, ...)
        settings    — numerical settings (grid sizes, tolerances, ...)
        config      — extra overrides (no tier enforcement)

    Method selection
        method           — global method tag (None → YAML default)
        method_overrides — per-stage overrides {(stage, target, scheme): tag}
        methods          — allowed method names (for validation)

    Run mode
        mode             — 'single', 'compare', or 'sweep'
        compare_methods  — method specs for compare mode

    Output
        output_dir — run output directory (auto-created)
        run_tag    — user-supplied tag (informational)

    Flags
        verbose, trace, mpi, gpu

    Simulation
        simulate, n_sim, seed, plots

    Sweep
        sweep_grids  — grid sizes for legacy --sweep-grids
        sweep_params — raw axis specs for --sweep-params
        sweep_runs   — best-of-n repetitions
        warmup       — JIT warmup before timing

    Experiment
        experiment_set, config_id, run_id

    Plot options
        skip_egm_plots, csv_export

    Extension
        extra — model-specific flags from extra_args
    """
    name: str
    model_dir: Path

    calib: dict
    settings: dict
    config: dict

    method: str | None
    method_overrides: dict
    methods: tuple

    mode: str
    compare_methods: tuple | None

    output_dir: Path
    run_tag: str | None

    verbose: bool
    trace: bool
    mpi: bool
    gpu: bool

    simulate: bool
    n_sim: int
    seed: int
    plots: bool

    sweep_grids: list | None
    sweep_params: list
    sweep_runs: int
    warmup: bool
    experiment_set: dict | None
    config_id: str | None
    run_id: str | None

    skip_egm_plots: bool
    csv_export: bool

    extra: dict

    def __getattr__(self, name):
        if name == 'syntax_dir':
            import warnings
            warnings.warn(
                "RunSpec.syntax_dir is deprecated; use RunSpec.model_dir",
                DeprecationWarning, stacklevel=2)
            return self.model_dir
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# parse_run — the single public entry point
# ---------------------------------------------------------------------------

def parse_run(
    name: str,
    syntax: str,
    methods: list[str],
    modes: list[str] | None = None,
    output: str = 'results',
    extra_args: dict | None = None,
) -> RunSpec:
    """Build a ``RunSpec`` from parsed command-line arguments.

    Parameters
    ----------
    name : human label used in ``--help``.
    syntax : path to the syntax directory (calibration.yaml, settings.yaml).
    methods : allowed method strings (e.g. ``['FUES', 'NEGM']``).
    modes : optional mode capabilities to register
        (any of ``'compare'``, ``'sweep'``, ``'simulate'``,
        ``'mpi'``, ``'gpu'``, ``'plots'``).
    output : default base output directory.
    extra_args : ``{flag: argparse_kwargs}`` for model-specific flags.

    If ``--method`` is omitted, ``RunSpec.method`` is ``None`` and the model
    YAML default selects the solution method.
    """
    modes = set(modes or [])

    # ---- parser ----------------------------------------------------------
    parser = argparse.ArgumentParser(description=f'Run {name} model')

    parser.add_argument(
        '--method', type=str, default=None, choices=methods,
        help='Solution method (omit to use YAML default)',
    )
    parser.add_argument(
        '--calib-override', nargs='+', action='append', default=[],
        help='Calibration param override: key=value (repeatable)',
    )
    parser.add_argument(
        '--setting-override', nargs='+', action='append', default=[],
        help='Settings override: key=value (repeatable)',
    )
    parser.add_argument(
        '--config-override', nargs='+', action='append', default=[],
        help='Config override: key=value (repeatable)',
    )
    parser.add_argument(
        '--override-file', type=str, default=None,
        help='YAML file with sparse overrides',
    )
    parser.add_argument(
        '--method-override', nargs='+', action='append', default=[],
        help='Method override: stage.target.scheme=TAG or stage.scheme=TAG (repeatable)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=output,
        help=f'Base output directory (default: {output})',
    )
    parser.add_argument('--run-tag', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--trace', action='store_true', default=False)

    if 'compare' in modes:
        parser.add_argument(
            '--compare', nargs='+', metavar='SPEC', default=None,
            help='Compare methods. Each SPEC is either a bare method name '
                 '(expanded via MODEL_SHORTCUT) or a method-override spec '
                 '(stage.scheme=TAG). Examples: '
                 '--compare FUES NEGM, '
                 '--compare keeper_cons.upper_envelope=FUES '
                 'keeper_cons.upper_envelope=MSS',
        )

    if 'sweep' in modes:
        parser.add_argument('--sweep', action='store_true', default=False)
        parser.add_argument(
            '--sweep-grids', type=str, default=None,
            help='Comma-separated grid sizes for sweep',
        )
        parser.add_argument(
            '--sweep-params', nargs='+', default=[],
            help='Sweep axes: key=val1,val2,... (Cartesian product). '
                 'Use method=FUES,NEGM to sweep methods.',
        )
        parser.add_argument('--sweep-runs', type=int, default=3)
        parser.add_argument('--warmup', '--no-warmup',
                            action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            '--experiment-set', type=str, default=None,
            help='YAML file defining experiment parameter sets',
        )
        parser.add_argument('--config-id', type=str, default=None)
        parser.add_argument('--run-id', type=str, default=None)

    if 'simulate' in modes:
        parser.add_argument('--simulate', action='store_true', default=False)
        parser.add_argument('--n-sim', type=int, default=10000)
        parser.add_argument('--seed', type=int, default=42)

    if 'mpi' in modes:
        parser.add_argument('--mpi', action='store_true', default=False)

    if 'gpu' in modes:
        parser.add_argument('--gpu', action='store_true', default=False)

    if 'plots' in modes:
        parser.add_argument('--plots', action='store_true', default=False)
        parser.add_argument('--skip-egm-plots', action='store_true', default=False)
        parser.add_argument('--csv-export', action='store_true', default=False)

    if extra_args:
        for flag, kwargs in extra_args.items():
            parser.add_argument(flag, **kwargs)

    args = parser.parse_args()

    # ---- resolve model dir -----------------------------------------------
    model_dir = Path(syntax).resolve()
    calib_path = model_dir / 'calibration.yaml'
    settings_path = model_dir / 'settings.yaml'

    # ---- load base YAML --------------------------------------------------
    base_calib = _load_yaml_section(calib_path, 'calibration')
    base_settings = _load_yaml_section(settings_path, 'settings')
    calib_keys = set(base_calib.keys())
    settings_keys = set(base_settings.keys())

    # ---- parse CLI overrides with tier enforcement -----------------------
    cli_calib = _parse_tiered_overrides(
        args.calib_override, 'calib-override', settings_keys, 'settings',
        own_keys=calib_keys)
    cli_settings = _parse_tiered_overrides(
        args.setting_override, 'setting-override', calib_keys, 'calibration',
        own_keys=settings_keys)
    cli_config = _parse_config_overrides(
        args.config_override, calib_keys, settings_keys)

    cli_method_overrides = {}
    for group in args.method_override:
        for item in group:
            key, tag = _parse_method_override_str(item)
            cli_method_overrides[key] = tag

    # ---- parse override file ---------------------------------------------
    file_calib, file_settings, file_config = {}, {}, {}
    if args.override_file:
        file_overrides = _load_override_file(args.override_file)
        for k, v in file_overrides.items():
            if k in calib_keys:
                file_calib[k] = v
            elif k in settings_keys:
                file_settings[k] = v
            else:
                warnings.warn(
                    f"Override-file key '{k}' not in calibration or settings; "
                    f"routing to config.")
                file_config[k] = v

    # ---- merge (precedence: CLI > override-file > base YAML) -------------
    calib = {**base_calib, **file_calib, **cli_calib}
    settings = {**base_settings, **file_settings, **cli_settings}
    config = {**file_config, **cli_config}

    # ---- determine mode --------------------------------------------------
    has_compare = getattr(args, 'compare', None) is not None
    has_sweep = getattr(args, 'sweep', False)

    active_modes = []
    if has_compare:
        active_modes.append('compare')
    if has_sweep:
        active_modes.append('sweep')
    if len(active_modes) > 1:
        parser.error(
            f"Mutually exclusive modes active: {', '.join(active_modes)}")

    if has_compare:
        mode = 'compare'
        compare_methods = tuple(args.compare)
        # Validate bare method names (not override specs) against choices
        for m in compare_methods:
            if '=' not in m and m not in methods:
                parser.error(
                    f"Invalid compare method '{m}'. "
                    f"Choose from: {', '.join(methods)} "
                    f"or use stage.scheme=TAG syntax")
    elif has_sweep:
        mode = 'sweep'
        compare_methods = None
    else:
        mode = 'single'
        compare_methods = None

    # ---- sweep grids parsing ---------------------------------------------
    sweep_grids = None
    raw_grids = getattr(args, 'sweep_grids', None)
    if raw_grids:
        sweep_grids = [int(x.strip()) for x in raw_grids.split(',')]

    # ---- experiment set loading ------------------------------------------
    experiment_set = None
    exp_path = getattr(args, 'experiment_set', None)
    if exp_path:
        with open(exp_path) as f:
            experiment_set = yaml.safe_load(f)

    # ---- output directory ------------------------------------------------
    output_dir = Path(make_run_dir(args.output_dir))

    # ---- collect extras --------------------------------------------------
    extra = {}
    if extra_args:
        for flag in extra_args:
            attr = flag.lstrip('-').replace('-', '_')
            extra[attr] = getattr(args, attr, None)

    # ---- build RunSpec ---------------------------------------------------
    return RunSpec(
        name=name,
        model_dir=model_dir,
        calib=calib,
        settings=settings,
        config=config,
        method=args.method,
        method_overrides=cli_method_overrides,
        methods=tuple(methods),
        mode=mode,
        compare_methods=compare_methods,
        output_dir=output_dir,
        run_tag=args.run_tag,
        verbose=args.verbose,
        trace=args.trace,
        mpi=getattr(args, 'mpi', False),
        gpu=getattr(args, 'gpu', False),
        simulate=getattr(args, 'simulate', False),
        n_sim=getattr(args, 'n_sim', 10000),
        seed=getattr(args, 'seed', 42),
        plots=getattr(args, 'plots', False),
        sweep_grids=sweep_grids,
        sweep_params=getattr(args, 'sweep_params', []),
        sweep_runs=getattr(args, 'sweep_runs', 3),
        warmup=getattr(args, 'warmup', True),
        experiment_set=experiment_set,
        config_id=getattr(args, 'config_id', None),
        run_id=getattr(args, 'run_id', None),
        skip_egm_plots=getattr(args, 'skip_egm_plots', False),
        csv_export=getattr(args, 'csv_export', False),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_method_override_str(raw: str) -> tuple[tuple[str, str, str], str]:
    """Parse 'stage.target.scheme=TAG' into ((stage, target, scheme), tag).

    Shorthand 'stage.scheme=TAG' defaults target to 'cntn_to_dcsn_builder'.
    """
    if '=' not in raw:
        raise ValueError(f"Method override must be path=TAG, got: {raw}")
    path, tag = raw.rsplit('=', 1)
    parts = path.split('.')
    if len(parts) == 3:
        stage, target, scheme = parts
    elif len(parts) == 2:
        stage, scheme = parts
        target = 'cntn_to_dcsn_builder'
    else:
        raise ValueError(
            f"Method override path must be stage.target.scheme or "
            f"stage.scheme, got: {path}")
    return (stage, target, scheme), tag.strip()


def _load_yaml_section(path: Path, section: str) -> dict:
    """Load a top-level section from a YAML file, returning flat dict."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get(section, {}))


def _parse_tiered_overrides(
    raw_groups: list[list[str]],
    source_name: str,
    forbidden_keys: set,
    forbidden_tier: str,
    own_keys: set | None = None,
) -> dict:
    """Parse nested ``nargs='+' action='append'`` lists and enforce tier."""
    result = {}
    for group in raw_groups:
        for item in group:
            k, v = _parse_kv(item)
            if k in forbidden_keys:
                raise SystemExit(
                    f"Error: --{source_name} key '{k}' belongs to "
                    f"{forbidden_tier}, not this tier.")
            if own_keys is not None and k not in own_keys:
                warnings.warn(
                    f"--{source_name} key '{k}' not found in "
                    f"calibration.yaml or settings.yaml.")
            result[k] = v
    return result


def _parse_config_overrides(
    raw_groups: list[list[str]],
    calib_keys: set,
    settings_keys: set,
) -> dict:
    """Parse --config-override with warnings for known-tier keys."""
    result = {}
    for group in raw_groups:
        for item in group:
            k, v = _parse_kv(item)
            if k in calib_keys:
                warnings.warn(
                    f"--config-override key '{k}' also exists in "
                    f"calibration.yaml (will override).")
            elif k in settings_keys:
                warnings.warn(
                    f"--config-override key '{k}' also exists in "
                    f"settings.yaml (will override).")
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Public helpers (kept for standalone use)
# ---------------------------------------------------------------------------

def parse_key_value(raw_list: list | None) -> dict:
    """Parse ``['key=value', ...]`` into ``{key: typed_value}``.

    Uses ``yaml.safe_load`` for type coercion (int, float, bool, str).

    >>> parse_key_value(['beta=0.96', 'grid_size=5000', 'verbose=true'])
    {'beta': 0.96, 'grid_size': 5000, 'verbose': True}
    """
    result = {}
    for item in (raw_list or []):
        k, v = _parse_kv(item)
        result[k] = v
    return result


def make_run_dir(base_dir, tag=None):
    """Create and return a run output directory.

    ``base_dir/YYYY-MM-DD/NNN/`` with auto-incremented NNN.
    Every run gets the next number.

    MPI-safe: rank 0 creates the directory and broadcasts the path to all
    ranks so every process uses the same folder (via ``kikku.run.mpi``).

    Example::

        results/durables/2026-03-25/001/
        results/durables/2026-03-25/002/
    """
    from .mpi import bcast_item, get_comm, rank_size

    if tag is not None:
        warnings.warn(
            "make_run_dir 'tag' parameter is deprecated and ignored. "
            "It will be removed in a future version.",
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
            run_dir = day_dir / f'{n:03d}'
            if not run_dir.exists():
                break
            n += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_str = str(run_dir)

    return bcast_item(run_dir_str, comm, root=0)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _parse_kv(item: str) -> tuple:
    """Split 'key=value' and coerce the value via yaml.safe_load."""
    if '=' not in item:
        raise ValueError(f"Override must be key=value, got: {item}")
    key, val = item.split('=', 1)
    return key.strip(), yaml.safe_load(val.strip())


def _load_override_file(path: str | Path) -> dict:
    """Load overrides from YAML file.

    Supports flat format or nested ``{'overrides': {...}}`` wrapper.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, dict) and 'overrides' in raw:
        return raw['overrides']
    return raw or {}
