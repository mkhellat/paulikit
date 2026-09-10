# cython: language_level=3
"""Cython binding for the fused coefficient kernel.

Replaces five separate NumPy passes over a (rows, dim) block - phase
factor, scale, threshold, index gather, dtype narrowing - with one C
pass that keeps each value in registers and appends survivors directly
to the output arrays. See ``coeffs.h`` for the design and the profile
that motivated it.

Optional, like the other extensions here: ``fwht.py`` falls back to
the NumPy stages if this module is absent, with identical results.

The GIL is released around the kernel call - it touches only the
buffers handed to it and no Python objects.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

cnp.import_array()

cdef extern from "coeffs.h":
    int64_t paulikit_coeffs_from_transformed(
        const double *data, int64_t rows, int64_t dim,
        const int64_t *row_x, double inv_dim, double atol,
        int index_bytes, void *out_x, void *out_z, double *out_coeff
    ) nogil


def coeffs_from_transformed(
    cnp.ndarray block,
    cnp.ndarray row_x,
    double atol,
    index_dtype=None,
):
    """Phase, scale, threshold and emit one transformed chunk.

    Args:
        block: C-contiguous ``(rows, dim)`` ``complex128`` array - the
            output of the Walsh-Hadamard transform. Read only.
        row_x: ``(rows,)`` integer array of the ``x`` bitmask for each
            row. Converted to ``int64`` if it is not already.
        atol: Terms with ``abs(coefficient) <= atol`` are dropped.
        index_dtype: dtype for the emitted ``x``/``z`` arrays -
            ``uint16``, ``uint32`` or ``int64``. Defaults to the
            narrowest width that can represent ``dim``, which is what
            removes the separate narrowing pass.

    Returns:
        ``(x, z, coefficient)`` - three 1-D arrays of equal length,
        already thresholded, in row-major order.

    Raises:
        ValueError: On a block that is not 2-D complex128 C-contiguous,
            a mismatched ``row_x`` length, a ``dim`` that is not a
            power of two, a negative ``atol``, or an ``index_dtype``
            too narrow to hold ``dim - 1``.
    """
    if block.ndim != 2:
        raise ValueError(f"expected a 2-D block, got {block.ndim}-D")
    if block.dtype != np.complex128:
        raise ValueError(f"expected complex128, got {block.dtype}")
    if not block.flags["C_CONTIGUOUS"]:
        raise ValueError("block must be C-contiguous")
    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}")

    cdef int64_t rows = block.shape[0]
    cdef int64_t dim = block.shape[1]
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of two, got {dim}")
    if row_x.shape[0] != rows:
        raise ValueError(
            f"row_x has {row_x.shape[0]} entries, block has {rows} rows"
        )

    if index_dtype is None:
        index_dtype = (
            np.uint16 if dim <= 1 << 16
            else np.uint32 if dim <= 1 << 32
            else np.int64
        )
    dt = np.dtype(index_dtype)
    cdef int index_bytes = dt.itemsize
    if index_bytes not in (2, 4, 8):
        raise ValueError(f"index_dtype must be 2, 4 or 8 bytes, got {dt}")
    # dim - 1 is the largest emittable z; x values are < dim too.
    if index_bytes < 8 and dim - 1 > np.iinfo(dt).max:
        raise ValueError(
            f"index_dtype {dt} cannot represent dim-1 = {dim - 1}"
        )

    cdef cnp.ndarray x_in = np.ascontiguousarray(row_x, dtype=np.int64)

    # Worst case: nothing is thresholded away. Bounded by
    # rows * dim - i.e. by chunk_size, never by dim**2 or by the total
    # term count - which is what keeps the pipeline's resident set
    # bounded. These buffers are smaller than the NumPy temporaries
    # they replace (640 KiB against ~2560 KiB at the N=150 shape).
    cdef int64_t capacity = rows * dim
    cdef cnp.ndarray out_x = np.empty(capacity, dtype=dt)
    cdef cnp.ndarray out_z = np.empty(capacity, dtype=dt)
    cdef cnp.ndarray out_c = np.empty(capacity, dtype=np.complex128)

    if rows == 0 or dim == 0:
        return out_x[:0], out_z[:0], out_c[:0]

    cdef const double *data = <const double *> cnp.PyArray_DATA(block)
    cdef const int64_t *xp = <const int64_t *> cnp.PyArray_DATA(x_in)
    cdef void *xo = cnp.PyArray_DATA(out_x)
    cdef void *zo = cnp.PyArray_DATA(out_z)
    cdef double *co = <double *> cnp.PyArray_DATA(out_c)
    cdef double inv_dim = 1.0 / <double> dim
    cdef int64_t n

    with nogil:
        n = paulikit_coeffs_from_transformed(
            data, rows, dim, xp, inv_dim, atol, index_bytes, xo, zo, co
        )

    # A slice keeps the WHOLE capacity buffer alive as its base, so a
    # sparse chunk would pin rows*dim entries to carry a handful of
    # terms. Bounded submission caps that at 2*n_workers chunks (~5 MiB
    # at the N=150 shape, against a ~65 MiB budget) so it is never
    # unsafe - but it is wasteful when most terms are thresholded away,
    # and the copy is cheap exactly when n is small. Copy below half
    # occupancy, slice above it: bounded either way, tighter when it
    # is free to be.
    if n * 2 < capacity:
        return (
            out_x[:n].copy(), out_z[:n].copy(), out_c[:n].copy()
        )
    return out_x[:n], out_z[:n], out_c[:n]
