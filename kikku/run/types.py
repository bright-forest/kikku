"""Frozen data shapes for RunSpec v2 (§12, kikku-runspec-v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TestSpec:
    """One test = one θ ∈ Θ.

    Carries override slot-fillings (relative to the base
    spec_factory's defaults) for the three groups. Each field is a
    dict {$ref: slot_value}, possibly empty if no overrides apply
    to that group for this test.

    label: short tag used by the output writer to identify the row
    in result tables. Auto-derived from non-empty fields if empty
    when the table is written.
    """

    params: dict[str, Any] = field(default_factory=dict)
    methods: dict[tuple[str, ...], str] | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class SimSpec:
    n_sim: int
    seed: int
    plots: bool


@dataclass(frozen=True)
class RunSpec:
    # Identity
    name: str
    base_spec: Path
    output_dir: Path
    run_tag: str | None

    # The test_set produced by make_test inside parse_cli.
    test_set: tuple[TestSpec, ...]

    # Key-membership sets, derived once from base_spec YAMLs at
    # parse time. Used by parse_cli for tier routing of CLI flags.
    params_keys: frozenset[str]
    settings_keys: frozenset[str]

    # Sweep meta
    sweep_runs: int
    warmup: bool

    # Simulation
    sim: SimSpec | None

    # Mode
    mode: str

    # Runner-specific knobs
    extra_args: dict

    # Flags
    verbose: bool
    trace: bool
    mpi: bool
    gpu: bool

    # Plot-export extras
    skip_egm_plots: bool
    csv_export: bool

    # Field-name aliases for runner code that read pre-v2 RunSpec fields.
    # REMOVE in v0.7.0 (or sooner if `grep -r "run.model_dir\|run.syntax_dir"`
    # in the FUES tree returns no hits in Steps 2-4).
    def __getattr__(self, name: str) -> object:
        if name == "model_dir":
            from warnings import warn

            warn(
                "RunSpec.model_dir is deprecated; use RunSpec.base_spec",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.base_spec
        if name == "syntax_dir":
            from warnings import warn

            warn(
                "RunSpec.syntax_dir is deprecated; use RunSpec.base_spec",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.base_spec
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}"
        )
