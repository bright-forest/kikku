"""Generic numerical utilities for DDSL stage operators.

Model-agnostic functions: clamping, multi-section golden
section maximization.  Used by both durables and retirement
operator factories.
"""

import numpy as np
from numba import njit


@njit
def clamp_value(arr, nan_val=-1e10):
    """Clamp NaN/inf in value arrays.

    NaN -> large negative so state is never chosen.
    """
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]) or np.isinf(result[i]):
            result[i] = nan_val
    return result


@njit
def clamp_policy(arr, min_val, max_val):
    """Clamp NaN/inf in policy arrays to valid bounds."""
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = min_val
        elif result[i] < min_val:
            result[i] = min_val
        elif result[i] > max_val or np.isinf(result[i]):
            result[i] = max_val
    return result


@njit
def clamp_scalar(val, min_val, max_val, nan_replacement):
    """Clamp a single scalar value.

    Returns nan_replacement if NaN.
    """
    if np.isnan(val):
        return nan_replacement
    elif np.isinf(val):
        return max_val if val > 0 else min_val
    elif val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    return val
