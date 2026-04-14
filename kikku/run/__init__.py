"""Runner infrastructure for DDSL model examples.

Provides everything an example's ``main()`` needs between ``solve()``
and disk output: CLI parsing, parameter sweeps, persistence, table
formatting, moment computation, and SMM estimation.

Modules
-------
cli       Parse command-line arguments into a frozen ``RunSpec``.
sweep     Parameter-grid sweep with best-of-n timing and MPI.
metrics   Markdown and LaTeX table formatting.
mpi       MPI distribution helpers (graceful degradation without mpi4py).
io        Solution save/load (numpy + JSON, legacy format).
nest_io   Nest save/load (.nst pickle files, dolo+ objects).
moments   Spec-driven moment generation (panels → dict[str, float]).
estimate  SMM estimation (cross-entropy optimizer, criterion composition).

Design
------
No model-specific code. No solve loops. No operators.
Each module has a simple interface that absorbs real complexity inside.
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
