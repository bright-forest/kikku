"""Dynx — structural layer above dolo-plus mod files.

Handles I/O, stage construction, period assembly, and
optional graph-based solve-order derivation.
"""

from .io import load_syntax
from .connectors import load_inter_connector
from .stage_maker import make_stage
from .period_maker import instantiate_period
from .graphs import period_to_graph, backward_paths, forward_order
from .methods import override_methods, parse_method_override_str

__all__ = [
    "load_syntax",
    "load_inter_connector",
    "make_stage",
    "instantiate_period",
    "period_to_graph",
    "backward_paths",
    "forward_order",
    "override_methods",
    "parse_method_override_str",
]
