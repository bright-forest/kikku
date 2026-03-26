"""Generic runner infrastructure for DDSL model examples.

Sub-modules:
    cli      — parse_run / RunSpec (CLI → frozen spec)
    sweep    — parameter sweep combinator (best-of-n, optional MPI)
    io       — solution save/load (numpy + json, legacy)
    nest_io  — nest save/load (.nst files, dolo+ objects + solutions)
    mpi      — MPI distribution helpers (graceful degradation)
    metrics  — table formatting (markdown + LaTeX)
    moments  — spec-driven moment generation (panels → dict[str, float])
    estimate — SMM estimation (CE, criterion composition, diagnostics)

No model-specific code. No solve loops. No operators.
"""

from .cli import parse_run, RunSpec, parse_key_value, make_run_dir
from .sweep import param_grid, sweep
from .io import save_solution, load_solution
from .nest_io import save_nest, load_nest, nest_info
from .mpi import get_comm, rank_size, is_root, scatter_items, gather_results, mpi_map, bcast_item
from .metrics import format_table, write_table
from .moments import (
    make_moment_fn, moment_names, load_data_moments,
    compute_moments_from_panels, compute_moments_from_dataframe,
)
from .estimate import (
    load_estimation_spec, make_criterion, estimate, diagnostics,
    EstimationResult,
)

__all__ = [
    # cli
    'parse_run', 'RunSpec', 'parse_key_value', 'make_run_dir',
    # sweep
    'param_grid', 'sweep',
    # io
    'save_solution', 'load_solution',
    # nest_io
    'save_nest', 'load_nest', 'nest_info',
    # mpi
    'get_comm', 'rank_size', 'is_root',
    'scatter_items', 'gather_results', 'mpi_map', 'bcast_item',
    # metrics
    'format_table', 'write_table',
    # moments
    'make_moment_fn', 'moment_names', 'load_data_moments',
    'compute_moments_from_panels', 'compute_moments_from_dataframe',
    # estimate
    'load_estimation_spec', 'make_criterion', 'estimate', 'diagnostics',
    'EstimationResult',
]
