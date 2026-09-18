# cython: language_level=3
"""Cython binding for the Hermiticity-violation check kernel.

Replaces `_check_hermitian_violation`'s four separate NumPy passes
(np.abs, np.abs, np.maximum, compare-and-any) with a single fused C
scan that stops at the first violation. See hermitian_check.h for the
design and the measurement that motivated it.

Optional, like the other extensions here: fwht.py falls back to the
NumPy 4-pass check if this module is absent, with identical results.

The GIL is released around the kernel call - it touches only the
buffer handed to it and no Python objects.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

cnp.import_array()

cdef extern from "hermitian_check.h":
    int64_t paulikit_first_hermitian_violation(
        const double *data, int64_t n, double atol
    ) nogil


def first_hermitian_violation(cnp.ndarray coefficient_values, double atol):
    """Index of the first coefficient whose imaginary part violates
    Hermiticity, or -1 if none.

    Args:
        coefficient_values: C-contiguous 1-D ``complex128`` array.
            Read only.
        atol: Same tolerance ``_check_hermitian_violation``'s Python
            reference takes.

    Returns:
        The 0-based index of the first violating element as a Python
        ``int``, or ``-1`` if none.

    Raises:
        ValueError: If ``coefficient_values`` is not 1-D complex128
            C-contiguous.
    """
    if coefficient_values.ndim != 1:
        raise ValueError(
            f"expected a 1-D array, got {coefficient_values.ndim}-D"
        )
    if coefficient_values.dtype != np.complex128:
        raise ValueError(f"expected complex128, got {coefficient_values.dtype}")
    if not coefficient_values.flags["C_CONTIGUOUS"]:
        raise ValueError("coefficient_values must be C-contiguous")

    cdef int64_t n = coefficient_values.shape[0]
    if n == 0:
        return -1

    cdef const double *data = <const double *> cnp.PyArray_DATA(coefficient_values)
    cdef int64_t result

    with nogil:
        result = paulikit_first_hermitian_violation(data, n, atol)

    return int(result)
