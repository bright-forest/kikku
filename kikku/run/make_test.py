"""Build TestSpec rows from pre-merged slot bundles (§5.5, v3)."""

from __future__ import annotations

from typing import Any

from .types import TestSpec


def make_test(slot_bundles: list[dict[str, Any]]) -> tuple[TestSpec, ...]:
    """Wrap each fully-merged slot bundle in a TestSpec; Cartesian work is in parse_cli."""
    if not slot_bundles:
        return (TestSpec(),)
    return tuple(TestSpec(slots=dict(b)) for b in slot_bundles)
