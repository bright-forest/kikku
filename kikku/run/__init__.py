"""Runner infrastructure for DDSL model examples (v3 RunSpec / parse_cli)."""

from .cli import parse_cli, parse_key_value, make_run_dir
from .make_test import make_test
from .sweep import SweepResult, param_grid, sweep
from .types import RunSpec, SimSpec, TestSpec
from .io import save_solution, load_solution
from .nest_io import save_nest, load_nest, nest_info
from .mpi import get_comm, rank_size, is_root, scatter_items, gather_results, mpi_map, bcast_item
from .metrics import format_table, write_table, write_results_table
from .moments import (
    make_moment_fn, moment_names, load_data_moments,
    compute_moments_from_panels, compute_moments_from_dataframe,
)
from .estimate import (
    load_estimation_spec, make_criterion, estimate, diagnostics,
    EstimationResult,
)

__all__ = [
    "parse_cli",
    "parse_key_value",
    "make_run_dir",
    "make_test",
    "RunSpec",
    "TestSpec",
    "SimSpec",
    "param_grid",
    "sweep",
    "SweepResult",
    "save_solution",
    "load_solution",
    "save_nest",
    "load_nest",
    "nest_info",
    "get_comm",
    "rank_size",
    "is_root",
    "scatter_items",
    "gather_results",
    "mpi_map",
    "bcast_item",
    "format_table",
    "write_table",
    "write_results_table",
    "make_moment_fn",
    "moment_names",
    "load_data_moments",
    "compute_moments_from_panels",
    "compute_moments_from_dataframe",
    "load_estimation_spec",
    "make_criterion",
    "estimate",
    "diagnostics",
    "EstimationResult",
]
