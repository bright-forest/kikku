"""Generic compose-and-reapproximate horse for arrival movers.

Given a decision-perch function defined on an irregular grid
(e.g. the endogenous EGM grid) and an arrival-to-decision
transition g, this horse composes f(g(x)) and evaluates it
on a Cartesian arrival grid via interpolation.

This is the computational core of the dcsn_to_arvl_mover:
it takes decision-perch arrays, maps the target grid through
g, and interpolates.
"""

from __future__ import annotations


def make_compose_interp(g_transition, interp_fn):
    """Build a compose-and-reapproximate callable.

    Parameters
    ----------
    g_transition : callable
        Maps arrival grid to decision grid::

            x_dcsn_targets = g_transition(x_arvl, params)

        Must be ``@njit``-compatible.  ``params`` is the
        model-specific scalar array.
    interp_fn : callable
        Fused multi-array interpolation function.  Signature
        depends on the number of arrays:

        - ``interp_as_2(xp, yp1, yp2, x) → (out1, out2)``
        - ``interp_as_3(xp, yp1, yp2, yp3, x) → (out1, out2, out3)``

        Both ``xp`` and ``x`` must be sorted ascending.

    Returns
    -------
    callable
        ``compose(x_dcsn_src, *y_arrays, x_arvl, params) → tuple``

        Applies ``g_transition(x_arvl, params)`` to get target
        points, then interpolates ``y_arrays`` from ``x_dcsn_src``
        onto those targets.
    """
    def compose(x_dcsn_src, *args):
        """Compose and interpolate.

        Last two positional args must be (x_arvl, params).
        All args before those are y-arrays to interpolate.
        """
        x_arvl = args[-2]
        params = args[-1]
        y_arrays = args[:-2]

        x_targets = g_transition(x_arvl, params)
        return interp_fn(x_dcsn_src, *y_arrays, x_targets)

    return compose
