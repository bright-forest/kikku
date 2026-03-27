"""Generic EGM backward movers (cntn_to_dcsn_mover).

Provides factory functions that accept sub-equation callables
(the "EGM recipe") and return compiled EGM operators:

*  ``make_egm_1d`` — scalar (1-D) continuation grid.
*  ``make_egm``    — K-dimensional continuation grid (pure EGM).

The callables follow a standardized signature::

    fn(pointwise_input..., fixed_state) → result

- ``fixed_state``: non-optimized state variables held constant
  during inversion (e.g. housing stock for a durables keeper;
  0.0 for a 1D consumption-savings model).
- Model parameters are baked into the callables at construction time.

The returned operators are ``@njit``-compiled and grid-agnostic:
the continuation grid is passed as an argument so the same
operator works with any grid size without JIT recompilation.

Theory reference
----------------
The K-dimensional factory (``make_egm``) implements *pure EGM*
(Dobrescu & Shanker, Theorem 4.1): the exogenous grid dimension
K equals the active-state dimension, and the Jacobian
non-degeneracy condition ensures local invertibility.  When
dim(active states) < dim(post-states), root-finding is needed
to construct the exogenous grid on a lower-dimensional
submanifold (Claim 3.2); that grid construction is outside this
module's scope — once the K-dim exogenous grid is formed, the
operator returned by ``make_egm`` can process it.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ── 1-D scalar EGM ────────────────────────────────────────────

def make_egm_1d(fn_inv_euler, fn_bellman_rhs, fn_cntn_to_dcsn,
                fn_concavity):
    """Build a 1D EGM backward operator from sub-equation callables.

    Parameters
    ----------
    fn_inv_euler : callable
        ``(dv_cntn_i, fixed_state) → c_i``
        Recover optimal control from marginal continuation value.
    fn_bellman_rhs : callable
        ``(c_i, v_cntn_i, fixed_state) → v_i``
        Reconstruct value at decision perch.
    fn_cntn_to_dcsn : callable
        ``(c_i, x_cntn_i, fixed_state) → x_dcsn_i``
        Build endogenous grid (the transition).
    fn_concavity : callable
        ``(c_i, ddv_cntn_i, fixed_state) → del_a_i``
        Concavity diagnostic for DCEGM upper envelope.

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
            c = fn_inv_euler(dv_cntn[i], fixed_state)
            v_hat[i] = fn_bellman_rhs(c, v_cntn[i],
                                      fixed_state)
            c_hat[i] = c
            x_dcsn_hat[i] = fn_cntn_to_dcsn(c, x_cntn[i],
                                             fixed_state)
            del_a[i] = fn_concavity(c, ddv_cntn[i],
                                    fixed_state)

        return c_hat, v_hat, x_dcsn_hat, del_a

    return egm_step


# ── K-dimensional EGM (pure EGM) ──────────────────────────────

def make_egm(fn_inv_euler, fn_bellman_rhs, fn_cntn_to_dcsn,
             fn_concavity, params, n_dims, n_controls):
    r"""Build a K-dimensional EGM backward operator (pure EGM).

    Generalises ``make_egm_1d`` to an arbitrary-dimensional
    continuation grid.  Given N points on a K-dimensional
    exogenous grid, the operator inverts the Euler equation at
    each point to recover the optimal control vector, then maps
    continuation-perch coordinates to decision-perch (endogenous)
    coordinates.

    Invertibility (Dobrescu & Shanker)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *  Pure EGM requires ``dim(active states) = dim(post-states)
       = K`` (Claim 3.1).  The Jacobian condition

       .. math::

           \det\,\partial_{x_{\mathrm{act}}}
               \operatorname{proj}\,E(x_{\mathrm{act}},\,\tilde e)
           \neq 0

       guarantees local invertibility (Theorem 4.1).
       ``fn_inv_euler`` encodes this: it must return a well-defined
       control for every gradient in the exogenous grid's image.

    *  Some grid dimensions may be **pass-through** (identity
       map from cntn to dcsn, e.g. housing stock H for a
       non-adjuster).  These dimensions are formally part of the
       K-dimensional grid but contribute trivially to the
       Jacobian (diagonal entry = 1).  The callable still
       receives their gradient and coordinate so it can use them
       in the inversion (e.g. ``inv_mu(dV/da', H)``).

    *  When ``dim(active) < dim(post-states)``, the exogenous grid
       lives on a lower-dimensional submanifold and root-finding
       is needed to construct it (Claim 3.2).  Once the grid is
       constructed, this operator can process it.

    *  For models with occasionally binding constraints, the
       exogenous grid is partitioned across constrained regions;
       each region is inverted separately.

    Parameters
    ----------
    fn_inv_euler : callable
        ``(dv_cntn_i: float[K], x_cntn_i: float[K], fixed_state,
        params) → c_i: float[n_c]``

        Recover optimal control vector from the K-dimensional
        gradient of the continuation value **and** the grid
        coordinate at one exogenous grid point.  The coordinate
        is needed when some grid dimensions enter the utility
        directly (e.g. housing stock H in ``inv_mu(dV/da', H)``).
    fn_bellman_rhs : callable
        ``(c_i: float[n_c], v_cntn_i: float, x_cntn_i: float[K],
        fixed_state, params) → v_i: float``

        Reconstruct the scalar value at the decision perch.
        Receives the grid coordinate because the period reward
        may depend on it (e.g. ``u(c, H)``).
    fn_cntn_to_dcsn : callable
        ``(c_i: float[n_c], x_cntn_i: float[K], fixed_state, params)
        → x_dcsn_i: float[K]``

        Map (control, continuation coordinate) to the K-dimensional
        endogenous grid coordinate.  Dimensions that are not
        inverted (pass-through states like H) return the input
        value unchanged: ``x_dcsn_i[k] = x_cntn_i[k]``.
    fn_concavity : callable
        ``(c_i: float[n_c], ddv_cntn_i: float[K], x_cntn_i: float[K],
        fixed_state, params) → del_a_i: float[K]``

        Concavity diagnostic per dimension for the upper-envelope
        step (e.g. FUES / RFC).  Pass-through dimensions can
        return 0.0.
    params : ndarray
        Model-specific scalar parameter array.
    n_dims : int
        Dimension K of the exogenous and endogenous grids.
    n_controls : int
        Number of control variables recovered by the inverse Euler.

    Returns
    -------
    callable
        ``egm_step(dv_cntn, ddv_cntn, v_cntn, x_cntn, fixed_state)``
        → ``(c_hat, v_hat, x_dcsn_hat, del_a)``

        Array shapes::

            dv_cntn     (N, K)      gradient of continuation value
            ddv_cntn    (N, K)      second-derivative diagonal
            v_cntn      (N,)        continuation values
            x_cntn      (N, K)      exogenous grid coordinates
            ─────────────────────────────────────────────────────
            c_hat       (N, n_c)    optimal controls
            v_hat       (N,)        decision-perch values
            x_dcsn_hat  (N, K)      endogenous grid coordinates
            del_a       (N, K)      concavity diagnostics

    Examples
    --------
    Housing non-adjuster with pass-through H (K = 2, n_c = 1):
        Grid is ``(a', H)``.  The inverse Euler uses both
        ``dv_cntn_i[0]`` (= ``dV/da'``) and ``x_cntn_i[1]``
        (= ``H``) to compute ``c = inv_mu(dV/da', H)``.
        The transition returns ``[c + a',  H]`` — the H
        dimension is identity.  This eliminates the explicit
        ``for i_h`` loop over housing levels.

    Retirement with two savings instruments (K = 2, n_c = 1):
        Two Euler equations; consumption recovered from one,
        no-arbitrage checked by the other.  Endogenous grid is
        2-D ``(w_fin, w_pen)``.

    Housing adjuster (K = 1, after root-finding):
        Active-state dimension is 1 (total resources) even though
        post-states are 2-D.  Root-finding first constructs the
        1-D exogenous grid on a submanifold of ``(a', h')``; then
        ``make_egm`` processes it with K = 1.
    """

    @njit
    def egm_step(dv_cntn, ddv_cntn, v_cntn, x_cntn, fixed_state):
        """EGM backward step over a K-dimensional continuation grid.

        Returns unrefined (pre-upper-envelope) decision-perch
        quantities on the endogenous grid.
        """
        n = x_cntn.shape[0]
        c_hat = np.zeros((n, n_controls))
        v_hat = np.zeros(n)
        x_dcsn_hat = np.zeros((n, n_dims))
        del_a = np.zeros((n, n_dims))

        for i in range(n):
            c = fn_inv_euler(dv_cntn[i], x_cntn[i],
                             fixed_state, params)
            v_hat[i] = fn_bellman_rhs(c, v_cntn[i], x_cntn[i],
                                      fixed_state, params)
            c_hat[i] = c
            x_dcsn_hat[i] = fn_cntn_to_dcsn(c, x_cntn[i],
                                             fixed_state, params)
            del_a[i] = fn_concavity(c, ddv_cntn[i], x_cntn[i],
                                    fixed_state, params)

        return c_hat, v_hat, x_dcsn_hat, del_a

    return egm_step
