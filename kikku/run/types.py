"""Frozen data shapes for RunSpec v3 (kikku-runspec-v3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TestSpec:
    """One test = one slot binding bundle (§3.1, v3).

    slots: slot name (no $ prefix) → slot value (mirrors YAML at that
    address). Empty dict means no override for any bound slot in this row.
    """

    slots: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class SimSpec:
    n_sim: int
    seed: int
    plots: bool


@dataclass(frozen=True)
class RunSpec:
    name: str
    base_spec: Path
    output_dir: Path
    run_tag: str | None

    test_set: tuple[TestSpec, ...]

    sweep_runs: int
    warmup: bool

    sim: SimSpec | None
    mode: str

    extra_args: dict

    verbose: bool
    trace: bool
    mpi: bool
    gpu: bool
    skip_egm_plots: bool
    csv_export: bool

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
