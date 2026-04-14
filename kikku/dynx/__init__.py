"""Dynx — re-export layer pointing at dolo-plus factory modules.

After spec 0.1r, all RCRK functions live in dolo.compiler.
This module re-exports old names for backward compatibility.
"""

from dolo.compiler.stage_factory.loader import load_syntax
from dolo.compiler.nest_factory.loader import load_inter_connector
from dolo.compiler.period_factory import make as make_period
from dolo.compiler.period_factory.graphs import period_to_graph, forward_order
from dolo.compiler.nest_factory.maker import backward_paths
from dolo.compiler.stage_factory import methodize, configure, calibrate

from .methods import override_methods, parse_method_override_str

# Backward-compat aliases
instantiate_period = make_period

__all__ = [
    "load_syntax",
    "load_inter_connector",
    "make_period",
    "instantiate_period",
    "period_to_graph",
    "backward_paths",
    "forward_order",
    "methodize",
    "configure",
    "calibrate",
    "override_methods",
    "parse_method_override_str",
]
