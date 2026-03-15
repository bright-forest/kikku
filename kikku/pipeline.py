"""Dolo-plus syntax loading and three-functor pipeline.

Two functions:

- ``load_syntax`` — I/O boundary: reads all YAML from disk.
- ``instantiate_period`` — pure transform: runs the pipeline
  on pre-loaded data.

Dolo-plus is imported lazily so that ``kikku.period_graphs``
remains usable without dolo-plus installed.
"""

from __future__ import annotations

import yaml
from pathlib import Path

from .nest import load_inter_connector


def load_syntax(syntax_dir, calib_overrides=None,
                config_overrides=None):
    """Load all YAML inputs from a dolo-plus syntax directory.

    All disk reads happen here.  Returns the raw inputs
    for each step of the pipeline.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory.
    calib_overrides, config_overrides : dict, optional
        Sparse overrides applied after loading.

    Returns
    -------
    calibration : dict
    settings : dict
    stage_sources : dict
        ``{name: {"yaml_text": ..., "yaml_path": ...,
        "methods": <loaded dict>}}``.
    period_template : dict
        ``{"name": ..., "stages": [...], "connectors": [...]}``.
    inter_conn : dict
        Inter-period connector from ``nest.yaml``.
    """
    try:
        from dolo.compiler.methodization import load_methodization
    except ImportError as e:
        raise ImportError(
            "load_syntax requires dolo-plus for loading "
            "methodization files."
        ) from e

    syntax_dir = Path(syntax_dir)

    with open(syntax_dir / "calibration.yaml") as f:
        calibration = yaml.safe_load(f)['calibration']
    if calib_overrides:
        calibration.update(calib_overrides)

    with open(syntax_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)['settings']
    if config_overrides:
        settings.update(config_overrides)

    with open(syntax_dir / "period.yaml") as f:
        raw = yaml.safe_load(f)
    stage_names = []
    for entry in raw.get('stages', []):
        if isinstance(entry, dict):
            stage_names.extend(entry.keys())
        else:
            stage_names.append(str(entry))

    stages_dir = syntax_dir / "stages"
    stage_sources = {}
    for name in stage_names:
        stage_yaml_path = stages_dir / name / f"{name}.yaml"
        with open(stage_yaml_path) as f:
            yaml_text = f.read()
        methods_path = stages_dir / name / f"{name}_methods.yml"
        methods_dict = load_methodization(methods_path)
        stage_sources[name] = {
            "yaml_text": yaml_text,
            "yaml_path": str(stage_yaml_path),
            "methods": methods_dict,
        }

    period_template = {
        "name": raw["name"],
        "stages": stage_names,
        "connectors": raw.get("connectors", []),
    }

    inter_conn = load_inter_connector(syntax_dir)

    return calibration, settings, stage_sources, \
        period_template, inter_conn


def instantiate_period(calibration, settings, stage_sources,
                       period_template):
    """Build one period via the dolo-plus pipeline.

    Applies ``parse → methodize → configure → calibrate``
    to every stage.  All inputs are pre-loaded data — no
    file paths, no disk I/O.

    Parameters
    ----------
    calibration : dict
        Economic parameters (consumed by ``calibrate``).
    settings : dict
        Numerical/structural settings (consumed by
        ``configure`` as a dict).
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
            "instantiate_period requires dolo-plus. "
            "Install it or use kikku.period_graphs directly "
            "with a pre-built period dict."
        ) from e

    params = {**calibration, **settings}

    stages = {}
    for name in period_template["stages"]:
        src = stage_sources[name]
        s = SymbolicModel(
            yaml.compose(src["yaml_text"]),
            filename=src["yaml_path"],
        )
        s = methodize_stage(s, src["methods"])
        s = configure_stage(s, settings)
        s = calibrate_stage(s, params)
        stages[name] = s

    return {
        "stages": stages,
        "connectors": period_template.get("connectors", []),
    }
