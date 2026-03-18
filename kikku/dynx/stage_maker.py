"""Per-stage functor chain: parse, methodize, configure, calibrate.

Placeholder for what will eventually be part of a revised dolo-plus.
"""

from __future__ import annotations


def make_stage(name, src, methods, calibration, settings):
    """Apply parse → methodize → configure → calibrate to one stage.

    Parameters
    ----------
    name : str
        Stage name (for error messages).
    src : dict
        ``{"yaml_text": str, "yaml_path": str}`` — pre-loaded
        YAML source for the stage.
    methods : dict
        Methodization dict (from ``*_methods.yml``).
    calibration : dict
        Economic parameters.
    settings : dict
        Numerical/structural settings.

    Returns
    -------
    SymbolicModel
        Fully calibrated stage object.

    Raises
    ------
    ImportError
        If dolo-plus is not installed.
    """
    try:
        import yaml
        from dolo.compiler.model import SymbolicModel
        from dolo.compiler.calibration import (
            calibrate as calibrate_stage,
            configure as configure_stage,
        )
        from dolo.compiler.methodization import methodize as methodize_stage
    except ImportError as e:
        raise ImportError(
            f"make_stage requires dolo-plus (stage: {name})."
        ) from e

    params = {**calibration, **settings}

    s = SymbolicModel(
        yaml.compose(src["yaml_text"]),
        filename=src["yaml_path"],
    )
    s = methodize_stage(s, methods)
    s = configure_stage(s, settings)
    s = calibrate_stage(s, params)
    return s
