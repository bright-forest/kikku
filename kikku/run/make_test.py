"""Build test_set = Cartesian product of param / method / setting ranges (§15)."""

from __future__ import annotations

import itertools
from typing import Any

from .types import TestSpec


def _finalize_methods(
    m: dict[tuple[str, ...], str] | None,
) -> dict[tuple[str, ...], str] | None:
    if m is None or len(m) == 0:
        return None
    return m


def make_test(
    *,
    params: list[dict[str, Any]] | None = None,
    methods: list[dict[tuple[str, ...], str] | None] | None = None,
    settings: list[dict[str, Any]] | None = None,
) -> tuple[TestSpec, ...]:
    """Build a test_set by Cartesian product across the three sweep
    dimensions. Each input is a list of test_spec_<group>
    alternatives (each a dict). None or empty inputs contribute
    a single default row in that dimension.

    Returns a tuple of TestSpecs of length:
        max(1, |params|) * max(1, |methods|) * max(1, |settings|).
    """
    p_alts: list[dict[str, Any]] = list(params) if params else [dict()]
    if not p_alts:
        p_alts = [dict()]

    s_alts: list[dict[str, Any]] = list(settings) if settings else [dict()]
    if not s_alts:
        s_alts = [dict()]

    if methods is None:
        m_alts: list[dict[tuple[str, ...], str] | None] = [None]
    elif not methods:
        m_alts = [None]
    else:
        m_alts = list(methods)

    out: list[TestSpec] = []
    for p, m, s in itertools.product(p_alts, m_alts, s_alts):
        out.append(
            TestSpec(
                params=dict(p),
                methods=_finalize_methods(m),
                settings=dict(s),
            )
        )
    return tuple(out)
