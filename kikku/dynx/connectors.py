"""Inter-period connector loading from nest.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_inter_connector(syntax_dir: str | Path) -> dict:
    """Load the inter-period connector from ``nest.yaml``.

    The connector maps continuation-perch poststates to
    arrival-perch prestates across period boundaries
    (spec 0.1h S3.3, unified connector terminology 0.1lA).

    ``nest.yaml`` uses custom YAML tags (``!period``).  This
    function absorbs that complexity so callers get a plain dict.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory containing ``nest.yaml``.

    Returns
    -------
    dict
        First ``inter_connectors`` entry (all identical for
        stationary models), e.g. ``{"b": "a", "b_ret": "a_ret"}``.
        Empty dict if no connectors are declared.
    """
    path = Path(syntax_dir) / "nest.yaml"

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor(
        "",
        lambda loader, suffix, node: (
            loader.construct_mapping(node)
            if isinstance(node, yaml.MappingNode)
            else None
        ),
    )

    with open(path) as f:
        nest_yaml = yaml.load(f, Loader=_Loader)

    connectors = nest_yaml.get("inter_connectors", [])
    if connectors:
        return connectors[0]
    return {}
