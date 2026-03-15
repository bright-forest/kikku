"""Generic 1D EGM backward mover (cntn_to_dcsn_mover).

Accepts four sub-equation callables (the "EGM recipe") and
returns a compiled EGM operator.  The callables follow a
standardized signature::

    fn(pointwise_input..., fixed_state, params) → scalar

- ``fixed_state``: non-optimized state variables held constant
  during inversion (e.g. housing stock for a durables keeper;
  0.0 for a 1D consumption-savings model).
- ``params``: model-specific scalar array (e.g. [beta, R, delta]).

The returned operator is ``@njit``-compiled and grid-agnostic:
the continuation grid is passed as an argument so the same
operator works with any grid size without JIT recompilation.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def make_egm_1d(fn_inv_euler, fn_bellman_rhs, fn_cntn_to_dcsn,
                fn_concavity, params):
    """Build a 1D EGM backward operator from sub-equation callables.

    Parameters
    ----------
    fn_inv_euler : callable
        ``(dv_cntn_i, fixed_state, params) → c_i``
        Recover optimal control from marginal continuation value.
    fn_bellman_rhs : callable
        ``(c_i, v_cntn_i, fixed_state, params) → v_i``
        Reconstruct value at decision perch.
    fn_cntn_to_dcsn : callable
        ``(c_i, x_cntn_i, fixed_state, params) → x_dcsn_i``
        Build endogenous grid (the transition).
    fn_concavity : callable
        ``(c_i, ddv_cntn_i, fixed_state, params) → del_a_i``
        Concavity diagnostic for DCEGM upper envelope.
    params : ndarray
        Model-specific scalar parameter array.

    Returns
    -------
    callable
        ``egm_step(dv_cntn, ddv_cntn, v_cntn, x_cntn, fixed_state)``
        → ``(c_hat, v_hat, x_dcsn_hat, del_a)``
    """

    @njit
    def egm_step(dv_cntn, ddv_cntn, v_cntn, x_cntn, fixed_state):
        """EGM backward step over the continuation grid.

        Returns unrefined (pre-upper-envelope) decision-perch
        quantities on the endogenous grid.
        """
        n = len(x_cntn)
        c_hat = np.zeros(n)
        v_hat = np.zeros(n)
        x_dcsn_hat = np.zeros(n)
        del_a = np.zeros(n)

        for i in range(n):
            c = fn_inv_euler(dv_cntn[i], fixed_state, params)
            v_hat[i] = fn_bellman_rhs(c, v_cntn[i],
                                      fixed_state, params)
            c_hat[i] = c
            x_dcsn_hat[i] = fn_cntn_to_dcsn(c, x_cntn[i],
                                             fixed_state, params)
            del_a[i] = fn_concavity(c, ddv_cntn[i],
                                    fixed_state, params)

        return c_hat, v_hat, x_dcsn_hat, del_a

    return egm_step
