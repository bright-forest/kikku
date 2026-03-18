"""Period assembly: loop make_stage over the stage list.

Produces the canonical period dict consumed by graphs.py
and by application-level solvers.
"""

from __future__ import annotations

from .stage_maker import make_stage


def instantiate_period(calibration, settings, stage_sources,
                       period_template):
    """Build one period by applying the dolo-plus pipeline to each stage.

    Parameters
    ----------
    calibration : dict
        Economic parameters (consumed by ``calibrate``).
    settings : dict
        Numerical/structural settings (consumed by ``configure``).
    stage_sources : dict
        Pre-loaded stage data per stage name::

            {name: {"yaml_text": str, "yaml_path": str,
                     "methods": dict}, ...}

    period_template : dict
        ``{"stages": [name, ...], "connectors": [...]}``.

    Returns
    -------
    dict
        Canonical period dict: ``{"stages": {...}, "connectors": [...]}``.
    """
    stages = {}
    for name in period_template["stages"]:
        src = stage_sources[name]
        stages[name] = make_stage(
            name,
            src={"yaml_text": src["yaml_text"],
                 "yaml_path": src["yaml_path"]},
            methods=src["methods"],
            calibration=calibration,
            settings=settings,
        )

    return {
        "stages": stages,
        "connectors": period_template.get("connectors", []),
    }
